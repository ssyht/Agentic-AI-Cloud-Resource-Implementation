"""
tests/test_tools.py — Unit tests for all 5 agent capabilities.

Run with:  pytest tests/ -v
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tools.tools import (
    fingerprint_workflow,
    get_live_pricing,
    run_multiobjective_optimization,
    constraint_filter,
    recommend_execution_plan,
    log_run_feedback,
    get_feedback_summary,
    decompose_workflow,
    recommend_profiling_run,
)


# ─── Fixtures ───────────────────────────────

@pytest.fixture
def neuron_pricing():
    raw = get_live_pricing.invoke({
        "providers": ["AWS", "GCP", "Azure"],
        "min_vcpu": 4,
        "min_ram_gb": 8,
        "exclude_gpu": True,
    })
    return raw

@pytest.fixture
def neuron_pareto(neuron_pricing):
    raw = run_multiobjective_optimization.invoke({
        "instances_json": neuron_pricing,
        "workflow_type": "neuron_simulation",
    })
    return raw


# ─── Fingerprinting ─────────────────────────

def test_fingerprint_neuron():
    result = json.loads(fingerprint_workflow.invoke({"workflow_description": "STDP neuron simulation using NEURON"}))
    assert result["workflow_type"] == "neuron_simulation"
    assert result["gpu_needed"] is False
    assert result["cpu_intensity"] == "high"

def test_fingerprint_rnaseq():
    result = json.loads(fingerprint_workflow.invoke({"workflow_description": "RNAseq gene expression with Seurat"}))
    assert result["workflow_type"] == "rnaseq"
    assert result["ram_intensity"] == "high"

def test_fingerprint_unknown():
    result = json.loads(fingerprint_workflow.invoke({"workflow_description": "molecular dynamics protein folding"}))
    assert result["workflow_type"] == "generic"


# ─── Pricing ────────────────────────────────

def test_pricing_returns_instances(neuron_pricing):
    instances = json.loads(neuron_pricing)
    assert len(instances) > 0
    assert all("price_hr" in i for i in instances)

def test_pricing_excludes_gpu():
    raw = get_live_pricing.invoke({"providers": ["AWS"], "min_vcpu": 1, "min_ram_gb": 1, "exclude_gpu": True})
    instances = json.loads(raw)
    assert all(not i["gpu"] for i in instances)

def test_pricing_provider_filter():
    raw = get_live_pricing.invoke({"providers": ["GCP"], "min_vcpu": 1, "min_ram_gb": 1, "exclude_gpu": False})
    instances = json.loads(raw)
    assert all(i["provider"] == "GCP" for i in instances)


# ─── Pareto Optimization ────────────────────

def test_pareto_returns_front(neuron_pareto):
    data = json.loads(neuron_pareto)
    assert "pareto_front" in data
    assert len(data["pareto_front"]) >= 1

def test_pareto_labels(neuron_pareto):
    data = json.loads(neuron_pareto)
    front = data["pareto_front"]
    labels = [c.get("label") for c in front]
    assert "COST_OPTIMAL" in labels
    assert "PERFORMANCE_OPTIMAL" in labels

def test_pareto_non_dominated(neuron_pareto):
    """No config in the Pareto front should be dominated by another."""
    front = json.loads(neuron_pareto)["pareto_front"]
    for i, c in enumerate(front):
        for j, other in enumerate(front):
            if i == j:
                continue
            dominated = (
                other["estimated_cost_per_run"] <= c["estimated_cost_per_run"] and
                other["estimated_runtime_hrs"] <= c["estimated_runtime_hrs"] and
                (other["estimated_cost_per_run"] < c["estimated_cost_per_run"] or
                 other["estimated_runtime_hrs"] < c["estimated_runtime_hrs"])
            )
            assert not dominated, f"Config {c['instance']} is dominated by {other['instance']}"


# ─── Implementation 1: Constraint Filter ────

def test_constraint_filter_respects_cost(neuron_pareto):
    result = json.loads(constraint_filter.invoke({
        "pareto_json": neuron_pareto,
        "max_cost_per_run": 0.80,
    }))
    for cfg in result["feasible_configurations"]:
        assert cfg["estimated_cost_per_run"] <= 0.80

def test_constraint_filter_respects_runtime(neuron_pareto):
    result = json.loads(constraint_filter.invoke({
        "pareto_json": neuron_pareto,
        "max_runtime_hrs": 1.5,
    }))
    for cfg in result["feasible_configurations"]:
        assert cfg["estimated_runtime_hrs"] <= 1.5

def test_constraint_filter_respects_provider(neuron_pareto):
    result = json.loads(constraint_filter.invoke({
        "pareto_json": neuron_pareto,
        "required_region": "AWS",
    }))
    for cfg in result["feasible_configurations"]:
        assert cfg["provider"] == "AWS"

def test_constraint_filter_impossible_returns_empty(neuron_pareto):
    """A $0.01/run budget should return no feasible configs."""
    result = json.loads(constraint_filter.invoke({
        "pareto_json": neuron_pareto,
        "max_cost_per_run": 0.01,
    }))
    assert len(result["feasible_configurations"]) == 0


# ─── Implementation 2: Execution Plan ───────

def test_execution_plan_neuron():
    result = json.loads(recommend_execution_plan.invoke({
        "workflow_type": "neuron_simulation",
        "chosen_instance_json": '{"instance": "c6i.4xlarge", "vcpu": 16, "ram_gb": 32, "price_hr": 0.68}',
    }))
    plan = result["execution_plan"]
    assert "topology" in plan
    assert "parallelism" in plan
    assert "checkpointing" in plan

def test_execution_plan_rnaseq_mentions_threads():
    result = json.loads(recommend_execution_plan.invoke({
        "workflow_type": "rnaseq",
        "chosen_instance_json": '{"instance": "r6i.4xlarge", "vcpu": 16, "ram_gb": 128, "price_hr": 1.008}',
    }))
    plan = result["execution_plan"]
    assert "16" in plan["parallelism"]  # vcpu count should appear in thread recommendation


# ─── Implementation 3: Adaptive Feedback ────

def test_feedback_logging():
    result = json.loads(log_run_feedback.invoke({
        "workflow_type": "neuron_simulation",
        "instance": "c6i.2xlarge",
        "vcpu": 8,
        "ram_gb": 16,
        "predicted_runtime_hrs": 2.1,
        "actual_runtime_hrs": 2.8,
        "success": True,
        "cpu_util": 0.88,
        "ram_util": 0.45,
        "user_accepted": True,
    }))
    assert result["feedback_logged"]["prediction_error_pct"] > 0
    assert result["total_feedback_records"] >= 1

def test_feedback_detects_high_error():
    result = json.loads(log_run_feedback.invoke({
        "workflow_type": "fastqc",
        "instance": "m6i.xlarge",
        "vcpu": 4,
        "ram_gb": 16,
        "predicted_runtime_hrs": 0.5,
        "actual_runtime_hrs": 1.5,  # 200% error
        "success": True,
        "cpu_util": 0.60,
        "ram_util": 0.30,
        "user_accepted": False,
    }))
    assert any("underestimated" in i or "overestimated" in i or "Prediction" in i
               for i in result["insights"])

def test_feedback_summary():
    result = json.loads(get_feedback_summary.invoke({"workflow_type": "neuron_simulation"}))
    assert "total_runs_logged" in result


# ─── Implementation 4: Workflow Decomposition ───

def test_decompose_rnaseq_stages():
    result = json.loads(decompose_workflow.invoke({
        "workflow_type": "rnaseq",
        "total_estimated_runtime_hrs": 4.0,
    }))
    stages = result["stage_resource_plan"]
    stage_names = [s["stage"] for s in stages]
    assert "alignment" in stage_names
    assert "preprocessing" in stage_names

def test_decompose_stage_durations_sum_to_total():
    total = 4.0
    result = json.loads(decompose_workflow.invoke({
        "workflow_type": "rnaseq",
        "total_estimated_runtime_hrs": total,
    }))
    summed = sum(s["duration_hrs"] for s in result["stage_resource_plan"])
    assert abs(summed - total) < 0.05  # allow small float rounding

def test_decompose_each_stage_has_recommendation():
    result = json.loads(decompose_workflow.invoke({
        "workflow_type": "neuron_simulation",
        "total_estimated_runtime_hrs": 2.0,
    }))
    for stage in result["stage_resource_plan"]:
        assert "recommended_type" in stage
        assert "example_aws" in stage


# ─── Implementation 5: Profiling Run ────────

def test_profiling_run_suggested(neuron_pareto):
    result = json.loads(recommend_profiling_run.invoke({
        "workflow_type": "neuron_simulation",
        "full_data_size_gb": 50.0,
        "pareto_json": neuron_pareto,
    }))
    assert result["sample_fraction_pct"] == 10
    assert result["estimated_profiling_cost_usd"] < 1.0  # profiling should be cheap
    assert len(result["observations_to_collect"]) > 0

def test_profiling_sample_capped_at_20gb(neuron_pareto):
    result = json.loads(recommend_profiling_run.invoke({
        "workflow_type": "rnaseq",
        "full_data_size_gb": 500.0,  # very large dataset
        "pareto_json": neuron_pareto,
    }))
    assert result["sample_data_size_gb"] <= 20.0

def test_profiling_sample_min_1gb(neuron_pareto):
    result = json.loads(recommend_profiling_run.invoke({
        "workflow_type": "fastqc",
        "full_data_size_gb": 2.0,  # small dataset, 10% = 0.2GB
        "pareto_json": neuron_pareto,
    }))
    assert result["sample_data_size_gb"] >= 1.0
