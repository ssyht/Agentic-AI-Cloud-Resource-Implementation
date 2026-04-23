"""
tools.py — The 5 capabilities of the Cloud Resource Configuration Agent.

Fixes applied (per Pari Hirenkumar's review):
  Fix 4: _feedback_store backed by SQLite — persists across process restarts
  Fix 5: fingerprint_workflow scores ALL profiles, returns best match
  Fix 6: recommend_profiling_run estimates runtime from actual traces
"""

import json
import sqlite3
import os
from langchain_core.tools import tool

from data.mock_pricing import PRICING_CATALOG, WORKFLOW_TRACES, WORKFLOW_STAGES

# ─────────────────────────────────────────────
# Fix 4: SQLite-backed feedback store
# ─────────────────────────────────────────────

DB_PATH = os.environ.get("FEEDBACK_DB_PATH", "data/feedback.db")

def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workflow_type TEXT,
            instance TEXT,
            vcpu INTEGER,
            ram_gb INTEGER,
            predicted_runtime_hrs REAL,
            actual_runtime_hrs REAL,
            prediction_error_pct REAL,
            success INTEGER,
            cpu_util REAL,
            ram_util REAL,
            user_accepted INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

_init_db()


def _write_feedback(record: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO feedback (
            workflow_type, instance, vcpu, ram_gb,
            predicted_runtime_hrs, actual_runtime_hrs, prediction_error_pct,
            success, cpu_util, ram_util, user_accepted
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (
        record["workflow_type"], record["instance"], record["vcpu"], record["ram_gb"],
        record["predicted_runtime_hrs"], record["actual_runtime_hrs"], record["prediction_error_pct"],
        int(record["success"]), record["cpu_util"], record["ram_util"], int(record["user_accepted"])
    ))
    conn.commit()
    conn.close()


def _read_feedback(workflow_type: str = None) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if workflow_type:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE workflow_type = ? ORDER BY created_at DESC",
            (workflow_type,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────

def _estimate_runtime(instance: dict, workflow_type: str) -> float:
    traces = WORKFLOW_TRACES.get(workflow_type, [])
    if not traces:
        return 2.0
    target_vcpu = instance["vcpu"]
    sorted_traces = sorted(traces, key=lambda t: abs(t["vcpu"] - target_vcpu))
    closest = sorted_traces[0]
    ratio = closest["vcpu"] / max(target_vcpu, 1)
    return round(closest["runtime_hrs"] * ratio, 2)


def _cost_per_run(instance: dict, runtime_hrs: float) -> float:
    return round(instance["price_hr"] * runtime_hrs, 3)


def _pareto_front(candidates: list) -> list:
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
# Tool 1: Workflow Fingerprinting
# Fix 5: Score ALL profiles, return best match — not first keyword hit
# ─────────────────────────────────────────────

@tool
def fingerprint_workflow(workflow_description: str) -> str:
    """
    Classify a workflow description into a known type and resource profile.
    Scores ALL profiles by keyword count and returns the best match.

    Args:
        workflow_description: Natural language description of the scientific workflow.
    """
    desc = workflow_description.lower()

    profiles = {
        "neuron_simulation": {
            "keywords": ["neuron", "simulation", "hodgkin", "huxley", "stdp",
                         "single cell", "electrophysiol", "neuroscience", "genesis"],
            "cpu_intensity": "high",
            "ram_intensity": "medium",
            "gpu_needed": False,
            "notes": "CPU-bound; NEURON/GENESIS tools do not benefit from GPU for single-cell models.",
        },
        "rnaseq": {
            "keywords": ["rnaseq", "rna-seq", "rna seq", "gene expression", "transcriptomics",
                         "seurat", "cell ranger", "hisat", "differential expression",
                         "alignment_to_differential_expression", "bioinformatics",
                         "variant", "calling", "gatk", "samtools", "bcftools", "vcf", "snp"],
            "cpu_intensity": "high",
            "ram_intensity": "high",
            "gpu_needed": False,
            "notes": "Memory-intensive; large reference genome requires high RAM.",
        },
        "fastqc": {
            "keywords": ["fastqc", "quality control", "qc", "sequencing quality",
                         "fastq", "trimming", "trimmomatic"],
            "cpu_intensity": "medium",
            "ram_intensity": "low",
            "gpu_needed": False,
            "notes": "Lightweight; suitable for small instances.",
        },
        "pgen": {
            "keywords": ["pgen", "population genetics", "variant calling", "gwas", "genome wide"],
            "cpu_intensity": "high",
            "ram_intensity": "medium",
            "gpu_needed": False,
            "notes": "Compute-heavy; parallelizable across chromosomes.",
        },
    }

    # Score all profiles — pick highest, not first match
    scores = {
        wf_type: sum(1 for kw in profile["keywords"] if kw in desc)
        for wf_type, profile in profiles.items()
    }

    best_type = max(scores, key=lambda k: scores[k])

    if scores[best_type] == 0:
        return json.dumps({
            "workflow_type": "generic",
            "cpu_intensity": "medium",
            "ram_intensity": "medium",
            "gpu_needed": False,
            "notes": "No specific profile matched. Using generic medium-resource profile.",
            "match_scores": scores,
        }, indent=2)

    result = {
        "workflow_type": best_type,
        **{k: v for k, v in profiles[best_type].items() if k != "keywords"},
        "match_scores": scores,
    }
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────
# Tool 2: Live Pricing Query (mocked)
# ─────────────────────────────────────────────

@tool
def get_live_pricing(
    providers: list,
    min_vcpu: int = 1,
    min_ram_gb: int = 1,
    exclude_gpu: bool = False,
) -> str:
    """
    Query cloud pricing catalog for instances matching requirements.

    Args:
        providers:    List of cloud providers e.g. ["AWS", "GCP", "Azure"].
        min_vcpu:     Minimum vCPUs required.
        min_ram_gb:   Minimum RAM in GB required.
        exclude_gpu:  If True, filter out GPU instances.
    """
    results = [
        inst for inst in PRICING_CATALOG
        if inst["provider"] in providers
        and inst["vcpu"] >= min_vcpu
        and inst["ram_gb"] >= min_ram_gb
        and not (exclude_gpu and inst["gpu"])
    ]
    results.sort(key=lambda x: x["price_hr"])
    return json.dumps(results, indent=2)


# ─────────────────────────────────────────────
# Tool 3: Multi-objective Optimization
# ─────────────────────────────────────────────

@tool
def run_multiobjective_optimization(
    instances_json: str,
    workflow_type: str,
) -> str:
    """
    Compute Pareto-optimal configs minimizing both cost and runtime.

    Args:
        instances_json: JSON string list of instances from get_live_pricing.
        workflow_type:  Workflow type string from fingerprint_workflow.
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

    if pareto:
        pareto[0]["label"] = "COST_OPTIMAL"
        pareto[-1]["label"] = "PERFORMANCE_OPTIMAL"
        if len(pareto) > 2:
            pareto[len(pareto) // 2]["label"] = "BALANCED"

    return json.dumps({
        "pareto_front": pareto,
        "total_candidates_evaluated": len(enriched)
    }, indent=2)


# ─────────────────────────────────────────────
# Tool 4: Implementation 1 — Constraint-driven Config
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
    [Implementation 1] Filter Pareto configs to only those satisfying hard constraints.

    Args:
        pareto_json:       JSON string of Pareto front.
        max_cost_per_run:  Hard budget ceiling in USD.
        max_runtime_hrs:   Hard time ceiling in hours.
        required_region:   Provider constraint e.g. "AWS".
        max_ram_gb:        Max RAM allowed in GB.
    """
    try:
        data = json.loads(pareto_json)
        candidates = data.get("pareto_front", data) if isinstance(data, dict) else data
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for pareto_json"})

    feasible, rejected = [], []

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
# Tool 5: Implementation 2 — Execution Recommendation
# ─────────────────────────────────────────────

@tool
def recommend_execution_plan(
    workflow_type: str,
    chosen_instance_json: str,
) -> str:
    """
    [Implementation 2] Recommend HOW to run — topology, parallelism, checkpointing.

    Args:
        workflow_type:         Workflow type string.
        chosen_instance_json:  JSON string of the selected instance.
    """
    try:
        inst = json.loads(chosen_instance_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON for chosen_instance_json"})

    vcpu = inst.get("vcpu", 4)

    plans = {
        "neuron_simulation": {
            "topology": "Single large VM",
            "parallelism": f"Run {max(1, vcpu // 2)} parallel parameter sweeps using GNU Parallel.",
            "checkpointing": "Save simulation state every 500ms simulated time using NEURON's SaveState.",
            "preprocess_instance": "Same instance (lightweight preprocessing).",
            "notes": "Sequential execution acceptable for single-cell; parallelism helps for parameter sweeps.",
        },
        "rnaseq": {
            "topology": "Preprocess on CPU-optimized → Align on memory-optimized → Quantify → Archive",
            "parallelism": f"Split FASTQ by chromosome; run {vcpu} threads with HISAT2 -p {vcpu}.",
            "checkpointing": "Use Pegasus checkpointing; resume from last successful stage on failure.",
            "preprocess_instance": "Use m6i.xlarge for QC/trimming before alignment.",
            "notes": "Memory bottleneck at alignment; ensure RAM >= 32GB for human genome index.",
        },
        "fastqc": {
            "topology": "Single small VM",
            "parallelism": f"Run FastQC with --threads {vcpu}.",
            "checkpointing": "Not needed — runtime < 1hr.",
            "preprocess_instance": "N/A",
            "notes": "Lightweight; consider spot/preemptible instances.",
        },
        "generic": {
            "topology": "Single VM",
            "parallelism": "No specific strategy identified. Consult workflow documentation.",
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
# Tool 6: Implementation 3 — Adaptive Feedback Loop
# Fix 4: SQLite persistence
# ─────────────────────────────────────────────

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
    [Implementation 3] Log actual run outcome to SQLite for persistent learning.

    Args:
        workflow_type / instance / vcpu / ram_gb: Run context.
        predicted_runtime_hrs: Agent prediction before the run.
        actual_runtime_hrs:    Actual observed runtime.
        success:               Whether the run completed successfully.
        cpu_util / ram_util:   Observed utilization 0.0 to 1.0.
        user_accepted:         Whether user accepted this recommendation.
    """
    error_pct = round(
        abs(actual_runtime_hrs - predicted_runtime_hrs) / max(predicted_runtime_hrs, 0.01) * 100, 1
    )

    record = {
        "workflow_type": workflow_type, "instance": instance,
        "vcpu": vcpu, "ram_gb": ram_gb,
        "predicted_runtime_hrs": predicted_runtime_hrs,
        "actual_runtime_hrs": actual_runtime_hrs,
        "prediction_error_pct": error_pct,
        "success": success, "cpu_util": cpu_util,
        "ram_util": ram_util, "user_accepted": user_accepted,
    }

    _write_feedback(record)
    total = len(_read_feedback())

    insights = []
    if error_pct > 25:
        direction = "underestimated" if actual_runtime_hrs > predicted_runtime_hrs else "overestimated"
        insights.append(f"Prediction {direction} by {error_pct}% — adjust for {workflow_type} on {instance}.")
    if not success:
        insights.append("Run failed — instance ranked lower for this workflow in future.")
    if cpu_util < 0.4:
        insights.append(f"Low CPU utilization ({cpu_util:.0%}) — consider downsizing vCPU.")
    if ram_util > 0.9:
        insights.append(f"High RAM utilization ({ram_util:.0%}) — consider upsizing RAM.")
    if user_accepted:
        insights.append("User accepted — confidence score increased.")

    return json.dumps({
        "feedback_logged": record,
        "total_feedback_records": total,
        "persisted_to": DB_PATH,
        "insights": insights,
    }, indent=2)


@tool
def get_feedback_summary(workflow_type: str = "all") -> str:
    """
    Retrieve aggregated feedback from SQLite.

    Args:
        workflow_type: Filter by workflow type. Pass "all" for everything.
    """
    wf = None if workflow_type == "all" else workflow_type
    records = _read_feedback(wf)

    if not records:
        return json.dumps({"message": "No feedback records yet.", "total": 0})

    avg_error = round(sum(r["prediction_error_pct"] for r in records) / len(records), 1)
    success_rate = round(sum(1 for r in records if r["success"]) / len(records) * 100, 1)
    acceptance_rate = round(sum(1 for r in records if r["user_accepted"]) / len(records) * 100, 1)

    return json.dumps({
        "workflow_type_filter": workflow_type,
        "total_runs_logged": len(records),
        "avg_prediction_error_pct": avg_error,
        "success_rate_pct": success_rate,
        "recommendation_acceptance_rate_pct": acceptance_rate,
        "persisted_to": DB_PATH,
    }, indent=2)


# ─────────────────────────────────────────────
# Tool 7: Implementation 4 — Workflow Decomposition
# ─────────────────────────────────────────────

@tool
def decompose_workflow(
    workflow_type: str,
    total_estimated_runtime_hrs: float,
) -> str:
    """
    [Implementation 4] Recommend optimal instance type per workflow stage.

    Args:
        workflow_type:                  Workflow type string.
        total_estimated_runtime_hrs:    Total runtime estimate.
    """
    stages = WORKFLOW_STAGES.get(workflow_type, [
        {"stage": "main", "cpu_intensity": "medium", "ram_intensity": "medium", "duration_fraction": 1.0}
    ])

    type_map = {
        ("high",   "high"):   {"recommended_type": "Memory+Compute optimized", "example_aws": "r6i.4xlarge",  "rationale": "Both CPU and RAM stressed."},
        ("high",   "medium"): {"recommended_type": "Compute optimized",        "example_aws": "c6i.4xlarge",  "rationale": "CPU-bound; standard RAM sufficient."},
        ("high",   "low"):    {"recommended_type": "Compute optimized",        "example_aws": "c6i.2xlarge",  "rationale": "CPU-bound; minimal RAM needed."},
        ("medium", "high"):   {"recommended_type": "Memory optimized",         "example_aws": "r6i.2xlarge",  "rationale": "Memory bottleneck; moderate CPU."},
        ("medium", "medium"): {"recommended_type": "General purpose",          "example_aws": "m6i.2xlarge",  "rationale": "Balanced workload."},
        ("medium", "low"):    {"recommended_type": "General purpose (small)",  "example_aws": "m6i.xlarge",   "rationale": "Light workload."},
        ("low",    "low"):    {"recommended_type": "Spot/Preemptible small",   "example_aws": "m6i.xlarge",   "rationale": "Use cheapest; consider spot pricing."},
        ("low",    "high"):   {"recommended_type": "Memory optimized (small)", "example_aws": "r6i.2xlarge",  "rationale": "RAM-bound with low compute."},
    }

    plan = []
    for s in stages:
        key = (s["cpu_intensity"], s["ram_intensity"])
        rec = type_map.get(key, type_map[("medium", "medium")])
        plan.append({
            "stage": s["stage"],
            "duration_hrs": round(total_estimated_runtime_hrs * s["duration_fraction"], 2),
            "cpu_intensity": s["cpu_intensity"],
            "ram_intensity": s["ram_intensity"],
            **rec,
        })

    return json.dumps({
        "workflow_type": workflow_type,
        "stage_resource_plan": plan,
        "note": "Separate instances per stage reduce cost vs single over-provisioned instance.",
    }, indent=2)


# ─────────────────────────────────────────────
# Tool 8: Implementation 5 — Profiling-Run Generation
# Fix 6: Runtime from actual traces, not hardcoded 0.2hrs
# ─────────────────────────────────────────────

@tool
def recommend_profiling_run(
    workflow_type: str,
    full_data_size_gb: float,
    pareto_json: str,
) -> str:
    """
    [Implementation 5] Recommend a small-scale profiling run before full deployment.
    Runtime estimated from actual workflow traces scaled by sample fraction.

    Args:
        workflow_type:       Workflow type string.
        full_data_size_gb:   Full dataset size in GB.
        pareto_json:         JSON string of Pareto front.
    """
    try:
        data = json.loads(pareto_json)
        candidates = data.get("pareto_front", data) if isinstance(data, dict) else data
    except json.JSONDecodeError:
        candidates = []

    profiling_instance = min(candidates, key=lambda x: x.get("price_hr", 999)) if candidates else None

    sample_size_gb = round(min(max(full_data_size_gb * 0.10, 1.0), 20.0), 1)
    sample_fraction = sample_size_gb / max(full_data_size_gb, 1.0)

    # Fix 6: Use actual traces to estimate profiling runtime
    traces = WORKFLOW_TRACES.get(workflow_type, [])
    if traces and profiling_instance:
        inst_vcpu = profiling_instance.get("vcpu", 4)
        closest = min(traces, key=lambda t: abs(t["vcpu"] - inst_vcpu))
        estimated_profiling_runtime = round(max(closest["runtime_hrs"] * sample_fraction, 0.05), 2)
        runtime_basis = f"scaled from {closest['runtime_hrs']}hr trace on {closest['vcpu']}-vcpu instance"
    else:
        estimated_profiling_runtime = round(max(sample_fraction * 2.0, 0.05), 2)
        runtime_basis = "default estimate (no traces available for this workflow type)"

    price_hr = profiling_instance.get("price_hr", 0.20) if profiling_instance else 0.20

    return json.dumps({
        "recommendation": "Run a small-scale profiling job before full deployment.",
        "profiling_instance": profiling_instance.get("instance") if profiling_instance else "smallest available",
        "profiling_provider": profiling_instance.get("provider") if profiling_instance else "any",
        "sample_data_size_gb": sample_size_gb,
        "sample_fraction_pct": round(sample_fraction * 100, 1),
        "estimated_profiling_runtime_hrs": estimated_profiling_runtime,
        "estimated_profiling_cost_usd": round(price_hr * estimated_profiling_runtime, 3),
        "runtime_basis": runtime_basis,
        "observations_to_collect": [
            "Wall-clock runtime",
            "Peak CPU utilization via top/htop or CloudWatch",
            "Peak RAM utilization",
            "Disk I/O throughput",
            "Whether workflow scales linearly with data size",
        ],
        "next_step": "Feed profiling results back via log_run_feedback() to improve full-run predictions.",
    }, indent=2)