"""
agent.py — Cloud Resource Configuration Agent

Fixes applied:
  Fix 1: Configurable LLM backend via AGENT_LLM_BACKEND env var
         Defaults to Anthropic Claude (matching DOME's stack)
         Supports "ollama" for local open-source model
  Fix 2: LangGraph removed — tools called directly, synchronous output
  Fix 3: Accepts structured input (pipeline_type, memory, cpu) from Notebook agent
         Skips fingerprint_workflow when structured input is available
  Fix 7: Returns structured AgentResult dict, not a raw printed string

Usage:
  # With Anthropic (default, matches DOME stack):
  export ANTHROPIC_API_KEY=sk-ant-...
  PYTHONPATH=. python3 agent/agent.py "variant calling on AWS, max $1.50/run"

  # With Ollama (local, no API key):
  export AGENT_LLM_BACKEND=ollama
  ollama serve && ollama pull llama3.1
  PYTHONPATH=. python3 agent/agent.py "variant calling on AWS, max $1.50/run"

  # From Notebook agent (structured input, skips fingerprinting):
  from agent.agent import run_agent
  result = run_agent(
      query="recommend cloud resources",
      pipeline_type="rnaseq",
      memory=64,
      cpu=16,
  )
"""

import os
import sys
import json

from tools.tools import (
    fingerprint_workflow,
    get_live_pricing,
    run_multiobjective_optimization,
    constraint_filter,
    recommend_execution_plan,
    decompose_workflow,
    recommend_profiling_run,
    log_run_feedback,
    get_feedback_summary,
)

# ─────────────────────────────────────────────
# Fix 1: Configurable LLM backend
# ─────────────────────────────────────────────

def _get_llm():
    """
    Returns the LLM client based on AGENT_LLM_BACKEND env var.
    Defaults to Anthropic Claude to match DOME's LLM stack.
    Set AGENT_LLM_BACKEND=ollama for local open-source model.
    """
    backend = os.environ.get("AGENT_LLM_BACKEND", "anthropic").lower()

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        print(f"[agent] Using Ollama backend: {model}")
        return ChatOllama(model=model, temperature=0)

    else:
        from langchain_anthropic import ChatAnthropic
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. "
                "Export it or set AGENT_LLM_BACKEND=ollama to use a local model."
            )
        print(f"[agent] Using Anthropic backend: {model}")
        return ChatAnthropic(model=model, api_key=api_key, temperature=0)


# ─────────────────────────────────────────────
# Fix 7: Structured AgentResult output schema
# ─────────────────────────────────────────────

def _build_agent_result(
    pipeline_type: str,
    recommended_instance: dict,
    feasible_configs: list,
    execution_plan: dict,
    constraints_applied: dict,
    summary: str,
) -> dict:
    """
    Structured output matching DOME's AgentResult pattern.
    All fields are extractable without parsing prose.
    """
    return {
        "agent": "cloud_resource_configuration",
        "pipeline_type": pipeline_type,
        "templates": [
            {
                "instance": cfg.get("instance"),
                "provider": cfg.get("provider"),
                "vcpu": cfg.get("vcpu"),
                "ram_gb": cfg.get("ram_gb"),
                "storage": cfg.get("storage"),
                "price_hr": cfg.get("price_hr"),
                "estimated_runtime_hrs": cfg.get("estimated_runtime_hrs"),
                "estimated_cost_per_run": cfg.get("estimated_cost_per_run"),
                "label": cfg.get("label", "FEASIBLE"),
            }
            for cfg in feasible_configs
        ],
        "recommended_template": {
            "instance": recommended_instance.get("instance"),
            "provider": recommended_instance.get("provider"),
            "vcpu": recommended_instance.get("vcpu"),
            "ram_gb": recommended_instance.get("ram_gb"),
            "price_hr": recommended_instance.get("price_hr"),
            "estimated_runtime_hrs": recommended_instance.get("estimated_runtime_hrs"),
            "estimated_cost_per_run": recommended_instance.get("estimated_cost_per_run"),
        },
        "compute_estimate": {
            "cpu": recommended_instance.get("vcpu"),
            "memory": recommended_instance.get("ram_gb"),
            "runtime_hrs": recommended_instance.get("estimated_runtime_hrs"),
            "cost_per_run_usd": recommended_instance.get("estimated_cost_per_run"),
        },
        "execution_plan": execution_plan,
        "constraints_applied": constraints_applied,
        "summary": summary,
        "status": "success" if feasible_configs else "no_feasible_configs",
    }


# ─────────────────────────────────────────────
# Fix 2: Direct tool calls — no LangGraph loop
# Fix 3: Accepts structured input from Notebook agent
# ─────────────────────────────────────────────

def run_agent(
    query: str,
    pipeline_type: str = None,
    memory: int = None,
    cpu: int = None,
    max_cost_per_run: float = None,
    max_runtime_hrs: float = None,
    required_provider: str = None,
    providers: list = None,
    decompose: bool = False,
    suggest_profiling: bool = False,
    full_data_size_gb: float = None,
) -> dict:
    """
    Main agent entry point. Returns a structured AgentResult dict.

    Fix 2: Calls tools directly — no LangGraph internal loop.
           DOME gets a synchronous result it can parse immediately.
    Fix 3: If pipeline_type/memory/cpu are provided by Notebook agent,
           fingerprint_workflow is skipped entirely.

    Args:
        query:             Natural language user query (always required).
        pipeline_type:     Structured input from Notebook agent — skips fingerprinting.
        memory:            RAM requirement from Notebook agent in GB.
        cpu:               CPU requirement from Notebook agent.
        max_cost_per_run:  Hard budget constraint in USD.
        max_runtime_hrs:   Hard time constraint in hours.
        required_provider: Lock to specific provider e.g. "AWS".
        providers:         List of providers to query. Defaults to all three.
        decompose:         If True, also return stage decomposition.
        suggest_profiling: If True, also return profiling run recommendation.
        full_data_size_gb: Dataset size for profiling recommendation.

    Returns:
        Structured AgentResult dict.
    """

    if providers is None:
        providers = ["AWS", "GCP", "Azure"]

    print(f"\n[agent] Query: {query}")

    # ── Step 1: Fingerprint (skip if Notebook already classified) ──────────
    if pipeline_type:
        # Fix 3: Use structured input from Notebook agent directly
        print(f"[agent] Step 1: Using Notebook pipeline_type={pipeline_type} (skipping fingerprint)")
        wf_type = pipeline_type
        gpu_needed = False
        min_vcpu = cpu if cpu else 1
        min_ram_gb = memory if memory else 1
    else:
        print("[agent] Step 1: Fingerprinting workflow from query...")
        fp_raw = fingerprint_workflow.invoke({"workflow_description": query})
        fp = json.loads(fp_raw)
        wf_type = fp["workflow_type"]
        gpu_needed = fp.get("gpu_needed", False)
        min_vcpu = cpu if cpu else 4
        min_ram_gb = memory if memory else 8
        print(f"[agent]         workflow_type={wf_type} | scores={fp.get('match_scores', {})}")

    # ── Step 2: Get pricing ────────────────────────────────────────────────
    print(f"[agent] Step 2: Querying pricing for {providers}...")
    pricing_raw = get_live_pricing.invoke({
        "providers": providers,
        "min_vcpu": min_vcpu,
        "min_ram_gb": min_ram_gb,
        "exclude_gpu": not gpu_needed,
    })

    # ── Step 3: Pareto optimization ────────────────────────────────────────
    print("[agent] Step 3: Running Pareto optimization...")
    pareto_raw = run_multiobjective_optimization.invoke({
        "instances_json": pricing_raw,
        "workflow_type": wf_type,
    })
    pareto_data = json.loads(pareto_raw)
    pareto_front = pareto_data.get("pareto_front", [])

    # ── Step 4: Apply constraints ──────────────────────────────────────────
    constraints = {}
    if max_cost_per_run: constraints["max_cost_per_run"] = max_cost_per_run
    if max_runtime_hrs:  constraints["max_runtime_hrs"] = max_runtime_hrs
    if required_provider: constraints["required_region"] = required_provider

    if constraints:
        print(f"[agent] Step 4: Applying constraints: {constraints}...")
        filter_raw = constraint_filter.invoke({
            "pareto_json": pareto_raw,
            **constraints,
        })
        filter_data = json.loads(filter_raw)
        feasible = filter_data.get("feasible_configurations", [])
        print(f"[agent]         {len(feasible)} feasible | {len(filter_data.get('rejected_configurations', []))} rejected")
    else:
        feasible = pareto_front
        print("[agent] Step 4: No constraints specified — all Pareto configs are feasible")

    if not feasible:
        return {
            "agent": "cloud_resource_configuration",
            "pipeline_type": wf_type,
            "templates": [],
            "recommended_template": None,
            "compute_estimate": None,
            "execution_plan": None,
            "constraints_applied": constraints,
            "summary": "No feasible configurations found matching the specified constraints.",
            "status": "no_feasible_configs",
        }

    best = feasible[0]

    # ── Step 5: Execution plan ─────────────────────────────────────────────
    print(f"[agent] Step 5: Building execution plan for {best['instance']}...")
    plan_raw = recommend_execution_plan.invoke({
        "workflow_type": wf_type,
        "chosen_instance_json": json.dumps(best),
    })
    plan_data = json.loads(plan_raw)
    execution_plan = plan_data.get("execution_plan", {})

    # ── Optional: Decomposition ────────────────────────────────────────────
    decomposition = None
    if decompose:
        print("[agent] Optional: Running workflow decomposition...")
        dec_raw = decompose_workflow.invoke({
            "workflow_type": wf_type,
            "total_estimated_runtime_hrs": best.get("estimated_runtime_hrs", 2.0),
        })
        decomposition = json.loads(dec_raw)

    # ── Optional: Profiling recommendation ────────────────────────────────
    profiling = None
    if suggest_profiling and full_data_size_gb:
        print("[agent] Optional: Generating profiling run recommendation...")
        prof_raw = recommend_profiling_run.invoke({
            "workflow_type": wf_type,
            "full_data_size_gb": full_data_size_gb,
            "pareto_json": pareto_raw,
        })
        profiling = json.loads(prof_raw)

    # ── Step 6: Build structured result ───────────────────────────────────
    summary = (
        f"Recommended {best['instance']} ({best['provider']}) for {wf_type} workflow. "
        f"Estimated cost: ${best['estimated_cost_per_run']}/run, "
        f"runtime: {best['estimated_runtime_hrs']}hrs. "
        f"{execution_plan.get('topology', '')}."
    )

    result = _build_agent_result(
        pipeline_type=wf_type,
        recommended_instance=best,
        feasible_configs=feasible,
        execution_plan=execution_plan,
        constraints_applied=constraints,
        summary=summary,
    )

    if decomposition:
        result["decomposition"] = decomposition
    if profiling:
        result["profiling_recommendation"] = profiling

    return result


# ─────────────────────────────────────────────
# CLI runner — pretty prints the structured result
# ─────────────────────────────────────────────

def _parse_cli_args():
    """Parse simple key=value flags from CLI args."""
    args = sys.argv[1:]
    query_parts = []
    kwargs = {}
    for arg in args:
        if "=" in arg:
            k, v = arg.split("=", 1)
            try:
                kwargs[k] = float(v)
            except ValueError:
                kwargs[k] = v
        else:
            query_parts.append(arg)
    return " ".join(query_parts), kwargs


if __name__ == "__main__":
    query, kwargs = _parse_cli_args()

    if not query:
        query = "What cloud setup do I need for a variant calling bioinformatics workflow on AWS?"

    result = run_agent(
        query=query,
        max_cost_per_run=kwargs.get("max_cost_per_run"),
        max_runtime_hrs=kwargs.get("max_runtime_hrs"),
        required_provider=kwargs.get("required_provider"),
        pipeline_type=kwargs.get("pipeline_type"),
        memory=int(kwargs["memory"]) if "memory" in kwargs else None,
        cpu=int(kwargs["cpu"]) if "cpu" in kwargs else None,
    )

    print("\n" + "=" * 60)
    print("AGENT RESULT (structured)")
    print("=" * 60)
    print(json.dumps(result, indent=2))