"""
agent.py — LangGraph ReAct Cloud Resource Configuration Agent

The agent follows the ReAct pattern (Reason → Act → Observe → Reason...)
as shown in the UI mockup from Sanjit's slides:
  THOUGHT → ACTION → OBS → ACTION → final response

The agent integrates all 5 implementations from the slides into one unified
reasoning loop. It decides which tools to call based on the user query.
"""

import os
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic

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

# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are the Cloud Resource Configuration Agent for the OnTimeRecommend science gateway system.

Your job is to help scientists (neuroscientists and bioinformaticians) configure optimal 
cloud infrastructure for their scientific workflows. You have five core capabilities:

1. CONSTRAINT-DRIVEN CONFIGURATION: Enforce hard user constraints (budget, time, region, RAM)
   before recommending configurations — not just preference ranking.

2. EXECUTION RECOMMENDATION: Recommend HOW to run the workflow (parallelism strategy,
   checkpointing, execution topology), not just which instance to use.

3. ADAPTIVE FEEDBACK LOOP: Learn from actual run outcomes. Log results and refine future
   predictions based on real CPU/RAM utilization, success rates, and user acceptance.

4. WORKFLOW DECOMPOSITION: Break complex workflows into stages and assign the right
   instance type per stage (e.g., cheap CPU for preprocessing, high-RAM for alignment).

5. PROFILING-RUN GENERATION: Before a full deployment, recommend a small-scale profiling 
   run to learn the workflow's actual behavior — especially for incomplete specs.

REASONING APPROACH:
- Always start by fingerprinting the workflow type.
- Then query pricing for matching instances.
- Run multi-objective optimization to find the Pareto front.
- Apply any user-specified constraints.
- Recommend an execution plan for the chosen config.
- If the user has large/expensive data, suggest a profiling run first.
- For complex multi-stage workflows, decompose by stage.
- Always explain your reasoning step by step (THOUGHT → ACTION → OBSERVATION).

Be concise but precise. Always report estimated cost per run AND estimated runtime.
For Pareto fronts, always show at least the COST_OPTIMAL and PERFORMANCE_OPTIMAL options.
"""

# ─────────────────────────────────────────────
# Build agent
# ─────────────────────────────────────────────

def build_agent():
    """Construct and return the LangGraph ReAct agent."""

    # Uses Claude claude-sonnet-4-20250514 — strong reasoning for multi-step tool use
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        temperature=0,  # deterministic for reproducibility in research
    )

    tools = [
        fingerprint_workflow,
        get_live_pricing,
        run_multiobjective_optimization,
        constraint_filter,
        recommend_execution_plan,
        log_run_feedback,
        get_feedback_summary,
        decompose_workflow,
        recommend_profiling_run,
    ]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent


# ─────────────────────────────────────────────
# Simple CLI runner
# ─────────────────────────────────────────────

def run_query(query: str, verbose: bool = True) -> str:
    """Run a single query through the agent and return the final response."""
    agent = build_agent()

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    final_message = result["messages"][-1].content

    if verbose:
        print("\n" + "=" * 60)
        print("AGENT RESPONSE")
        print("=" * 60)
        print(final_message)

    return final_message


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What cloud setup do I need to run STDP neuron simulations at scale? Budget: $2 per run max."

    run_query(query)
