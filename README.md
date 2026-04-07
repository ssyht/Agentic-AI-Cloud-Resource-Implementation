# OnTimeRecommend — Cloud Resource Configuration Agent

**Part of the OnTimeRecommend science gateway recommender system**  
University of Missouri-Columbia · NSF OAC-1730655 / OAC-2006816

---

## Overview

This repository implements the **Cloud Resource Configuration Agent** — an agentic upgrade to the Cloud Solution Template Recommender described in the OnTimeRecommend paper (Vekaria et al., CCPE 2020).

The original system used a static 300-instance KNN + ILP approach. This agent replaces it with a **LangGraph ReAct agent** that reasons step-by-step (THOUGHT → ACTION → OBSERVE) and integrates five capabilities:

| # | Capability | Addresses limitation in original |
|---|---|---|
| 1 | **Constraint-driven configuration** | Arbitrary Red/Green/Gold tiers with undefined thresholds |
| 2 | **Execution recommendation** | Cost returned but no runtime or topology advice |
| 3 | **Adaptive feedback loop** | No learning from actual run outcomes |
| 4 | **Workflow decomposition** | One workflow = one configuration |
| 5 | **Profiling-run generation** | No handling of incomplete workflow specs |

---

## Repository Structure

```
ontimerecommend-cloud-agent/
├── agent/
│   └── agent.py          ← LangGraph ReAct agent (main entry point)
├── tools/
│   └── tools.py          ← All 5 capabilities as LangChain tools
├── data/
│   └── mock_pricing.py   ← Mock cloud pricing + workflow traces
├── metrics/
│   └── metrics.py        ← One metric per implementation (for the paper)
├── tests/
│   └── test_tools.py     ← 24 unit tests (one per tool behavior)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone <your-repo-url>
cd ontimerecommend-cloud-agent
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run the agent with a query
python agent/agent.py "What cloud setup do I need to run RNAseq analysis? Budget: $5 per run."

# 4. Run all unit tests
pytest tests/ -v

# 5. Generate metrics report (for the paper)
python metrics/metrics.py
```

---

## Code Walkthrough (for Ashish)

### How the agent reasons

The agent follows the **ReAct pattern** — the same flow shown in Sanjit's UI mockup:

```
User query
    ↓
THOUGHT: "This is a neuron simulation — CPU-bound, no GPU needed."
ACTION:  fingerprint_workflow("STDP neuron simulation")
OBS:     {workflow_type: "neuron_simulation", cpu_intensity: "high", gpu_needed: false}
    ↓
THOUGHT: "I need to find matching instances and compute Pareto-optimal configs."
ACTION:  get_live_pricing(providers=["AWS","GCP","Azure"], exclude_gpu=True)
ACTION:  run_multiobjective_optimization(instances, "neuron_simulation")
OBS:     Pareto front with COST_OPTIMAL ($0.34/hr, 2.1hrs) and PERFORMANCE_OPTIMAL ($0.76/hr, 0.9hrs)
    ↓
THOUGHT: "User has a $2/run budget. Apply constraint filter."
ACTION:  constraint_filter(pareto, max_cost_per_run=2.0)
OBS:     2 feasible configs, 1 rejected
    ↓
THOUGHT: "Recommend how to run the chosen config."
ACTION:  recommend_execution_plan("neuron_simulation", chosen_instance)
    ↓
Final response with Pareto configs + execution plan
```

### The 5 tools (Implementations 1–5)

#### Implementation 1 — `constraint_filter()`
**Location:** `tools/tools.py` → `constraint_filter()`  
Enforces hard user constraints (budget ceiling, runtime deadline, cloud region, max RAM) and returns only **feasible** configurations. Unlike the original Red/Green/Gold tiers, thresholds are **user-specified**, not arbitrary.

```python
constraint_filter(
    pareto_json=pareto_output,
    max_cost_per_run=1.50,      # hard budget
    max_runtime_hrs=2.0,        # hard deadline
    required_region="AWS",      # provider lock-in requirement
)
```

#### Implementation 2 — `recommend_execution_plan()`
**Location:** `tools/tools.py` → `recommend_execution_plan()`  
Returns an **execution topology** for the chosen instance — parallelism strategy (e.g., `HISAT2 -p 16`), checkpointing approach (e.g., Pegasus resume), and whether to use a single VM or split pre/post processing.

```python
recommend_execution_plan(
    workflow_type="rnaseq",
    chosen_instance_json='{"instance": "r6i.4xlarge", "vcpu": 16, ...}'
)
```

#### Implementation 3 — `log_run_feedback()` + `get_feedback_summary()`
**Location:** `tools/tools.py` → `log_run_feedback()`  
After each actual workflow run, logs the outcome (predicted vs actual runtime, CPU/RAM utilization, success/failure, user acceptance). The agent uses this to detect systematic over/under-estimation and adjust future rankings.

```python
log_run_feedback(
    workflow_type="neuron_simulation",
    predicted_runtime_hrs=2.1,
    actual_runtime_hrs=2.8,
    cpu_util=0.88, ram_util=0.45,
    success=True, user_accepted=True,
)
```

#### Implementation 4 — `decompose_workflow()`
**Location:** `tools/tools.py` → `decompose_workflow()`  
Splits the workflow into stages (preprocessing, alignment, simulation, archival) and recommends a different **instance type per stage** — e.g., cheap general-purpose for preprocessing, memory-optimized for alignment.

```python
decompose_workflow(
    workflow_type="rnaseq",
    total_estimated_runtime_hrs=4.0
)
# Returns: stage-by-stage resource plan with per-stage instance recommendations
```

#### Implementation 5 — `recommend_profiling_run()`
**Location:** `tools/tools.py` → `recommend_profiling_run()`  
Before committing to a full expensive run, suggests a **10% data sample** on the cheapest Pareto instance to observe actual CPU/RAM/disk behavior. Costs < 1% of a failed full run for large datasets.

```python
recommend_profiling_run(
    workflow_type="rnaseq",
    full_data_size_gb=200.0,
    pareto_json=pareto_output
)
```

---

## Metrics (for the paper)

Run `python metrics/metrics.py` to reproduce these results:

| Metric | Definition | Result |
|--------|-----------|--------|
| **M1: Constraint Satisfaction Rate** | % of returned configs satisfying ALL hard constraints | **100%** |
| **M2: Execution Plan Coverage** | % of plan fields populated (non-generic) | **94%** |
| **M3: Prediction Error Reduction** | % reduction in runtime error after feedback | **90.9%** |
| **M4: Decomposition Cost Savings** | % cost saved vs. single-instance baseline | **17–44%** |
| **M5: Profiling Cost Efficiency** | Ratio of full-run cost to profiling-run cost | **87x** |

---

## Replacing Mock Data with Live APIs

The mock pricing in `data/mock_pricing.py` is designed for easy swap-out:

```python
# Current (mock):
PRICING_CATALOG = [{"provider": "AWS", "instance": "c6i.2xlarge", ...}]

# Replace with live AWS pricing:
import boto3
client = boto3.client("pricing", region_name="us-east-1")
response = client.get_products(ServiceCode="AmazonEC2", ...)

# Replace with live GCP pricing:
from google.cloud import billing_v1
client = billing_v1.CloudCatalogClient()
skus = client.list_skus(parent="services/6F81-5844-456A")
```

---

## Connection to OnTimeRecommend Architecture

This agent plugs into the existing OnTimeRecommend middleware as a drop-in replacement for the Cloud Solution Template Recommender microservice:

```
Vidura Chatbot
    ↓ REST API call
OnTimeRecommend Middleware (Recommender Factory)
    ↓
Cloud Resource Configuration Agent  ← THIS REPO
    ↓
Pareto-optimal configs + execution plan + stage decomposition
    ↓
User on CyNeuro / KBCommons portal
```

The agent exposes the same REST interface as the original recommender (see Table 3 in the paper), so no changes are needed to the Vidura chatbot or the science gateway UI.
