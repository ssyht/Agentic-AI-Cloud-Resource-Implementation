"""
tools.py — The 5 capabilities of the Cloud Resource Configuration Agent.

Each function is a LangGraph tool the agent can call during reasoning.

Implementation mapping (from Sanjit's slides):
  1. fingerprint_workflow()         → workload fingerprinting classifier
  2. get_live_pricing()             → live cloud pricing API (mocked)
  3. run_multiobjective_opt()       → Pareto-optimal config selection
  4. constraint_filter()           → Implementation 1: Constraint-driven configuration
  5. recommend_execution_plan()    → Implementation 2: Execution recommendation
  6. log_run_feedback()            → Implementation 3: Adaptive feedback loop
  7. decompose_workflow()          → Implementation 4: Workflow decomposition
  8. recommend_profiling_run()     → Implementation 5: Profiling-run generation
"""

import json
import math
from typing import Any
from langchain_core.tools import tool

from data.mock_pricing import PRICING_CATALOG, WORKFLOW_TRACES, WORKFLOW_STAGES

# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _estimate_runtime(instance: dict, workflow_type: str) -> float:
    """Estimate runtime in hours based on historical traces for this workflow type."""
    traces = WORKFLOW_TRACES.get(workflow_type, [])
    if not traces:
        return 2.0  # default fallback

    # Find closest match by vcpu
    target_vcpu = instance["vcpu"]
    sorted_traces = sorted(traces, key=lambda t: abs(t["vcpu"] - target_vcpu))
    closest = sorted_traces[0]

    # Scale runtime inversely with vcpu ratio
    ratio = closest["vcpu"] / max(target_vcpu, 1)
    estimated = closest["runtime_hrs"] * ratio
    return round(estimated, 2)


def _cost_per_run(instance: dict, runtime_hrs: float) -> float:
    return round(instance["price_hr"] * runtime_hrs, 3)


def _pareto_front(candidates: list[dict]) -> list[dict]:
    """Return Pareto-optimal configs (minimize cost AND runtime — neither dominates the other)."""
    pareto = []
    for c in candidates:
        dominated = False
        for other in candidates:
            if other is c:
                continue
            if (other["estimated_cost_per_run"] <= c["estimated_cost_per_run"] and
                    other["estimated_runtime_hrs"] <= c["estimated_runtime_hrs"] and
                    (other["estimated_cost_per_run"] < c["estimated_cost_per_run"] or
                     other["estimated_runtime_hrs"] < c["estimated_runtime_hrs"])):
                dominated = True
                break
        if not dominated:
            pareto.append(c)
    return pareto


# ─────────────────────────────────────────────
# Tool 1 — Workflow Fingerprinting
# ─────────────────────────────────────────────

@tool
def fingerprint_workflow(workflow_description: str) -> str:
    """
    Classify a workflow description into a known type and resource profile.
    Returns workflow_type, cpu_intensity, ram_intensity, gpu_needed, and notes.

    Args:
        workflow_description: Natural language description of the scientific workflow.
    """
    desc = workflow_description.lower()

    profiles = {
        "neuron_simulation": {
            "keywords": ["neuron", "simulation", "hodgkin", "huxley", "stdp", "single cell", "electrophysiol"],
            "cpu_intensity": "high",
            "ram_intensity": "medium",
            "gpu_needed": False,
            "notes": "CPU-bound; NEURON/GENESIS tools do not benefit from GPU for single-cell models.",
        },
        "rnaseq": {
            "keywords": ["rnaseq", "rna-seq", "rna seq", "gene expression", "transcriptomics", "alignment", "seurat", "cell ranger"],
            "cpu_intensity": "high",
            "ram_intensity": "high",
            "gpu_needed": False,
            "notes": "Memory-intensive; large reference genome requires high RAM.",
        },
        "fastqc": {
            "keywords": ["fastqc", "quality control", "qc", "sequencing quality", "fastq"],
            "cpu_intensity": "medium",
            "ram_intensity": "low",
            "gpu_needed": False,
            "notes": "Lightweight; suitable for small instances.",
        },
        "pgen": {
            "keywords": ["pgen", "population genetics", "variant", "gwas"],
            "cpu_intensity": "high",
            "ram_intensity": "medium",
            "gpu_needed": False,
            "notes": "Compute-heavy; parallelizable across chromosomes.",
        },
    }

    matched_type = "generic"
    matched_profile = {
        "cpu_intensity": "medium",
        "ram_intensity": "medium",
        "gpu_needed": False,
        "notes": "No specific profile matched. Using generic medium-resource profile.",
    }

    for wf_type, profile in profiles.items():
        if any(kw in desc for kw in profile["keywords"]):
            matched_type = wf_type
            matched_profile = profile
            break

    result = {"workflow_type": matched_type, **matched_profile}
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────
# Tool 2 — Live Pricing Query (mocked)
# ─────────────────────────────────────────────

@tool
def get_live_pricing(
    providers: list[str],
    min_vcpu: int = 1,
    min_ram_gb: int = 1,
    exclude_gpu: bool = False,
) -> str:
    """
    Query cloud pricing catalog for instances matching requirements.
    Currently uses a mock catalog; swap for boto3/GCP/Azure SDK calls when credentials are available.

    Args:
        providers:    List of cloud providers to include, e.g. ["AWS", "GCP", "Azure"].
        min_vcpu:     Minimum number of virtual CPUs required.
        min_ram_gb:   Minimum RAM in GB required.
        exclude_gpu:  If True, filter out GPU instances.
    """
    results = []
    for inst in PRICING_CATALOG:
        if inst["provider"] not in providers:
            continue
        if inst["vcpu"] < min_vcpu:
            continue
        if inst["ram_gb"] < min_ram_gb:
            continue
        if exclude_gpu and inst["gpu"]:
            continue
        results.append(inst)

    results.sort(key=lambda x: x["price_hr"])
    return json.dumps(results, indent=2)


# ─────────────────────────────────────────────
# Tool 3 — Multi-objective Optimization (Pareto front)
# ─────────────────────────────────────────────

@tool
def run_multiobjective_optimization(
    instances_json: str,
    workflow_type: str,
) -> str:
    """
    Given a list of candidate instances (as JSON string from get_live_pricing),
    compute estimated runtime and cost-per-run for each, then return the
    Pareto-optimal set (minimizing both cost and runtime simultaneously).

    Args:
        instances_json: JSON string list of instance dicts from get_live_pricing.
        workflow_type:  The workflow type string from fingerprint_workflow.
    """
    try:
        instances = json.loads(instances_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for instances_json"})

    enriched = []
    for inst in instances:
        rt = _estimate_runtime(inst, workflow_type)
        cost = _cost_per_run(inst, rt)
        enriched.append({**inst, "estimated_runtime_hrs": rt, "estimated_cost_per_run": cost})

    pareto = _pareto_front(enriched)
    pareto.sort(key=lambda x: x["estimated_cost_per_run"])

    # Label the front
    if pareto:
        pareto[0]["label"] = "COST_OPTIMAL"
        pareto[-1]["label"] = "PERFORMANCE_OPTIMAL"
        if len(pareto) > 2:
            mid = len(pareto) // 2
            pareto[mid]["label"] = "BALANCED"

    return json.dumps({"pareto_front": pareto, "total_candidates_evaluated": len(enriched)}, indent=2)


# ─────────────────────────────────────────────
# Tool 4 — Implementation 1: Constraint-driven Configuration
# ─────────────────────────────────────────────

@tool
def constraint_filter(
    pareto_json: str,
    max_cost_per_run: float = None,
    max_runtime_hrs: float = None,
    required_region: str = None,
    max_ram_gb: int = None,
) -> str:
    """
    [Implementation 1: Constraint-driven Configuration]
    Filter Pareto-optimal configs to only those satisfying hard user constraints.
    Unlike preference-based ranking, this enforces strict feasibility boundaries.

    Args:
        pareto_json:       JSON string of Pareto front from run_multiobjective_optimization.
        max_cost_per_run:  Hard budget ceiling in USD per run (e.g., 1.50).
        max_runtime_hrs:   Hard time ceiling in hours (e.g., 2.0).
        required_region:   Cloud provider constraint (e.g., "AWS" to restrict to AWS only).
        max_ram_gb:        Max RAM allowed (e.g., for data privacy or licensing reasons).
    """
    try:
        data = json.loads(pareto_json)
        candidates = data.get("pareto_front", data) if isinstance(data, dict) else data
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for pareto_json"})

    feasible = []
    rejected = []

    for c in candidates:
        reasons = []
        if max_cost_per_run is not None and c["estimated_cost_per_run"] > max_cost_per_run:
            reasons.append(f"cost ${c['estimated_cost_per_run']} > limit ${max_cost_per_run}")
        if max_runtime_hrs is not None and c["estimated_runtime_hrs"] > max_runtime_hrs:
            reasons.append(f"runtime {c['estimated_runtime_hrs']}h > limit {max_runtime_hrs}h")
        if required_region is not None and c["provider"] != required_region:
            reasons.append(f"provider {c['provider']} != required {required_region}")
        if max_ram_gb is not None and c["ram_gb"] > max_ram_gb:
            reasons.append(f"RAM {c['ram_gb']}GB > limit {max_ram_gb}GB")

        if reasons:
            rejected.append({**c, "rejected_because": reasons})
        else:
            feasible.append(c)

    return json.dumps({
        "feasible_configurations": feasible,
        "rejected_configurations": rejected,
        "constraints_applied": {
            "max_cost_per_run": max_cost_per_run,
            "max_runtime_hrs": max_runtime_hrs,
            "required_region": required_region,
            "max_ram_gb": max_ram_gb,
        }
    }, indent=2)


# ─────────────────────────────────────────────
# Tool 5 — Implementation 2: Execution Recommendation
# ─────────────────────────────────────────────

@tool
def recommend_execution_plan(
    workflow_type: str,
    chosen_instance_json: str,
) -> str:
    """
    [Implementation 2: Execution Recommendation]
    Given a workflow type and chosen instance, recommend HOW to run it —
    not just where. Covers parallelism strategy, checkpointing, and execution topology.

    Args:
        workflow_type:         The workflow type string from fingerprint_workflow.
        chosen_instance_json:  JSON string of the selected instance config.
    """
    try:
        inst = json.loads(chosen_instance_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for chosen_instance_json"})

    vcpu = inst.get("vcpu", 4)
    ram = inst.get("ram_gb", 16)

    plans = {
        "neuron_simulation": {
            "topology": "Single large VM",
            "parallelism": f"Run {max(1, vcpu // 2)} parallel parameter sweeps using GNU Parallel or NEURON's built-in batch mode.",
            "checkpointing": "Save simulation state every 500ms simulated time using NEURON's SaveState.",
            "preprocess_instance": "Same instance (lightweight preprocessing).",
            "notes": "Sequential execution acceptable for single-cell; parallelism helps for parameter sweeps.",
        },
        "rnaseq": {
            "topology": "Preprocess on CPU-optimized → Align on memory-optimized → Quantify → Archive to cold storage",
            "parallelism": f"Split FASTQ by chromosome; run {vcpu} alignment threads with HISAT2 -p {vcpu}.",
            "checkpointing": "Use Pegasus workflow checkpointing; resume from last successful stage on failure.",
            "preprocess_instance": "Use a smaller instance (e.g., m6i.xlarge) for QC/trimming before alignment.",
            "notes": "Memory bottleneck is at alignment; ensure RAM ≥ 32GB for human genome index.",
        },
        "fastqc": {
            "topology": "Single small VM",
            "parallelism": f"Run FastQC with --threads {vcpu} flag.",
            "checkpointing": "Not needed — runtime < 1hr.",
            "preprocess_instance": "N/A",
            "notes": "Lightweight; consider spot/preemptible instances to reduce cost.",
        },
        "generic": {
            "topology": "Single VM",
            "parallelism": "No specific parallelism strategy identified. Consult workflow documentation.",
            "checkpointing": "Recommended for runtimes > 2hrs.",
            "preprocess_instance": "N/A",
            "notes": "Profile the workload before scaling.",
        },
    }

    plan = plans.get(workflow_type, plans["generic"])
    return json.dumps({
        "workflow_type": workflow_type,
        "selected_instance": inst.get("instance", "unknown"),
        "execution_plan": plan,
    }, indent=2)


# ─────────────────────────────────────────────
# Tool 6 — Implementation 3: Adaptive Feedback Loop
# ─────────────────────────────────────────────

# In-memory store (replace with a DB in production)
_feedback_store: list[dict] = []

@tool
def log_run_feedback(
    workflow_type: str,
    instance: str,
    vcpu: int,
    ram_gb: int,
    predicted_runtime_hrs: float,
    actual_runtime_hrs: float,
    success: bool,
    cpu_util: float,
    ram_util: float,
    user_accepted: bool,
) -> str:
    """
    [Implementation 3: Adaptive Feedback Loop]
    Log the outcome of an actual workflow run. The agent uses this data to:
      - Compute prediction error
      - Flag systematic over/under-estimation
      - Personalize future rankings
      - Build workload-specific confidence scores

    Args:
        workflow_type:         Workflow type string.
        instance:              Instance name that was used.
        vcpu / ram_gb:         Actual resources used.
        predicted_runtime_hrs: What the agent predicted before the run.
        actual_runtime_hrs:    What actually happened.
        success:               Whether the run completed successfully.
        cpu_util / ram_util:   Observed utilization (0.0 to 1.0).
        user_accepted:         Whether the user accepted this recommendation.
    """
    error_pct = round(abs(actual_runtime_hrs - predicted_runtime_hrs) / max(predicted_runtime_hrs, 0.01) * 100, 1)

    record = {
        "workflow_type": workflow_type,
        "instance": instance,
        "vcpu": vcpu,
        "ram_gb": ram_gb,
        "predicted_runtime_hrs": predicted_runtime_hrs,
        "actual_runtime_hrs": actual_runtime_hrs,
        "prediction_error_pct": error_pct,
        "success": success,
        "cpu_util": cpu_util,
        "ram_util": ram_util,
        "user_accepted": user_accepted,
    }
    _feedback_store.append(record)

    # Derive insights
    insights = []
    if error_pct > 25:
        direction = "underestimated" if actual_runtime_hrs > predicted_runtime_hrs else "overestimated"
        insights.append(f"Prediction {direction} by {error_pct}% — model will adjust for {workflow_type} on {instance}.")
    if not success:
        insights.append("Run failed — this instance will be ranked lower for this workflow in future recommendations.")
    if cpu_util < 0.4:
        insights.append(f"Low CPU utilization ({cpu_util:.0%}) — consider downsizing vCPU for this workflow.")
    if ram_util > 0.9:
        insights.append(f"High RAM utilization ({ram_util:.0%}) — consider upsizing RAM for this workflow.")
    if user_accepted:
        insights.append("User accepted this recommendation — confidence score increased.")

    return json.dumps({
        "feedback_logged": record,
        "total_feedback_records": len(_feedback_store),
        "insights": insights,
    }, indent=2)


@tool
def get_feedback_summary(workflow_type: str = None) -> str:
    """
    Retrieve aggregated feedback statistics to show adaptive learning over time.

    Args:
        workflow_type: Optional filter. If None, returns stats for all workflows.
    """
    records = [r for r in _feedback_store if workflow_type is None or r["workflow_type"] == workflow_type]

    if not records:
        return json.dumps({"message": "No feedback records yet.", "total": 0})

    avg_error = round(sum(r["prediction_error_pct"] for r in records) / len(records), 1)
    success_rate = round(sum(1 for r in records if r["success"]) / len(records) * 100, 1)
    acceptance_rate = round(sum(1 for r in records if r["user_accepted"]) / len(records) * 100, 1)

    return json.dumps({
        "workflow_type_filter": workflow_type or "all",
        "total_runs_logged": len(records),
        "avg_prediction_error_pct": avg_error,
        "success_rate_pct": success_rate,
        "recommendation_acceptance_rate_pct": acceptance_rate,
        "records": records,
    }, indent=2)


# ─────────────────────────────────────────────
# Tool 7 — Implementation 4: Workflow Decomposition
# ─────────────────────────────────────────────

@tool
def decompose_workflow(
    workflow_type: str,
    total_estimated_runtime_hrs: float,
) -> str:
    """
    [Implementation 4: Workflow Decomposition]
    Instead of recommending a single instance for the whole workflow,
    decompose the workflow into stages and recommend the optimal instance
    type for each stage (e.g., cheap CPU for preprocessing, high-RAM for alignment).

    Args:
        workflow_type:                  Workflow type string from fingerprint_workflow.
        total_estimated_runtime_hrs:    Total runtime estimate for the full workflow.
    """
    stages = WORKFLOW_STAGES.get(workflow_type, [
        {"stage": "main", "cpu_intensity": "medium", "ram_intensity": "medium", "duration_fraction": 1.0}
    ])

    # Instance type recommendations per intensity combo
    type_map = {
        ("high",   "high"):   {"recommended_type": "Memory+Compute optimized", "example_aws": "r6i.4xlarge",   "example_gcp": "n2-highmem-8",    "rationale": "Both CPU and RAM are stressed."},
        ("high",   "medium"): {"recommended_type": "Compute optimized",        "example_aws": "c6i.4xlarge",   "example_gcp": "c2-standard-16",   "rationale": "CPU-bound; standard RAM sufficient."},
        ("high",   "low"):    {"recommended_type": "Compute optimized",        "example_aws": "c6i.2xlarge",   "example_gcp": "c2-standard-8",    "rationale": "CPU-bound; minimal RAM needed."},
        ("medium", "high"):   {"recommended_type": "Memory optimized",         "example_aws": "r6i.2xlarge",   "example_gcp": "n2-highmem-8",    "rationale": "Memory bottleneck; moderate CPU."},
        ("medium", "medium"): {"recommended_type": "General purpose",          "example_aws": "m6i.2xlarge",   "example_gcp": "n2-standard-4",   "rationale": "Balanced workload."},
        ("medium", "low"):    {"recommended_type": "General purpose (small)",  "example_aws": "m6i.xlarge",    "example_gcp": "n2-standard-4",   "rationale": "Light workload."},
        ("low",    "low"):    {"recommended_type": "Spot/Preemptible small",   "example_aws": "m6i.xlarge",    "example_gcp": "n2-standard-4",   "rationale": "Use cheapest available; consider spot pricing."},
        ("low",    "high"):   {"recommended_type": "Memory optimized (small)", "example_aws": "r6i.2xlarge",   "example_gcp": "n2-highmem-8",   "rationale": "RAM-bound with low compute."},
    }

    plan = []
    for s in stages:
        key = (s["cpu_intensity"], s["ram_intensity"])
        rec = type_map.get(key, type_map[("medium", "medium")])
        stage_runtime = round(total_estimated_runtime_hrs * s["duration_fraction"], 2)
        plan.append({
            "stage": s["stage"],
            "duration_hrs": stage_runtime,
            "cpu_intensity": s["cpu_intensity"],
            "ram_intensity": s["ram_intensity"],
            **rec,
        })

    return json.dumps({
        "workflow_type": workflow_type,
        "stage_resource_plan": plan,
        "note": "Separate instances per stage can reduce cost vs. over-provisioning a single large instance.",
    }, indent=2)


# ─────────────────────────────────────────────
# Tool 8 — Implementation 5: Profiling-Run Generation
# ─────────────────────────────────────────────

@tool
def recommend_profiling_run(
    workflow_type: str,
    full_data_size_gb: float,
    pareto_json: str,
) -> str:
    """
    [Implementation 5: Profiling-Run Generation]
    Before committing to a full deployment, suggest a small-scale profiling run
    to learn the workflow's actual resource behavior. The agent intelligently
    selects a profiling instance and data subsample size.

    Args:
        workflow_type:       Workflow type string.
        full_data_size_gb:   Size of the full dataset in GB.
        pareto_json:         JSON string of Pareto front for reference.
    """
    try:
        data = json.loads(pareto_json)
        candidates = data.get("pareto_front", data) if isinstance(data, dict) else data
    except json.JSONDecodeError:
        candidates = []

    # Pick the cheapest candidate for the profiling run
    profiling_instance = None
    if candidates:
        profiling_instance = min(candidates, key=lambda x: x.get("price_hr", 999))

    # Recommend 10% data sample, min 1GB, max 20GB
    sample_size_gb = round(min(max(full_data_size_gb * 0.10, 1.0), 20.0), 1)
    estimated_profiling_runtime = round(0.10 * 2.0, 2)  # ~10% of typical 2hr run

    observations_to_collect = [
        "Wall-clock runtime",
        "Peak CPU utilization (via top/htop or CloudWatch)",
        "Peak RAM utilization",
        "Disk I/O throughput",
        "Whether the workflow scales linearly with data size",
    ]

    return json.dumps({
        "recommendation": "Run a small-scale profiling job before full deployment.",
        "profiling_instance": profiling_instance.get("instance") if profiling_instance else "Use smallest available instance",
        "profiling_provider": profiling_instance.get("provider") if profiling_instance else "Any",
        "sample_data_size_gb": sample_size_gb,
        "sample_fraction_pct": 10,
        "estimated_profiling_runtime_hrs": estimated_profiling_runtime,
        "estimated_profiling_cost_usd": round(
            (profiling_instance.get("price_hr", 0.20) if profiling_instance else 0.20) * estimated_profiling_runtime, 3
        ),
        "observations_to_collect": observations_to_collect,
        "next_step": "Feed profiling results back via log_run_feedback() to improve full-run predictions.",
    }, indent=2)
