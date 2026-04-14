"""
agent.py — LangGraph ReAct Cloud Resource Configuration Agent
Uses Ollama (local, free, no API key needed) instead of Anthropic API.

Setup (one time):
  1. Install Ollama:   brew install ollama
  2. Start Ollama:     ollama serve
  3. Pull the model:   ollama pull llama3.1
  4. Run the agent:    PYTHONPATH=. python3 agent/agent.py "your query here"

Why llama3.1?
  Best open source model for multi-step tool calling.
  Runs fully locally on Mac — no internet, no API key, no cost.
"""

import sys
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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

SYSTEM_PROMPT = """You are a cloud resource configuration agent for the OnTimeRecommend science gateway.

When a user asks about cloud setup for a scientific workflow, follow these steps IN ORDER and complete ALL of them:

Step 1: Call fingerprint_workflow with the workflow description to classify the workflow type.
Step 2: Call get_live_pricing with providers as a list like ["AWS","GCP","Azure"] and set exclude_gpu based on the fingerprint result.
Step 3: Call run_multiobjective_optimization. Pass the COMPLETE JSON string output from get_live_pricing as instances_json, and the workflow_type string from fingerprint_workflow as workflow_type.
Step 4: If the user mentioned a budget limit, time limit, or cloud provider preference, call constraint_filter with the pareto results.
Step 5: Call recommend_execution_plan for the best configuration from the Pareto front.
Step 6: Write a clear final recommendation including instance name, cost per run, estimated runtime, and execution advice.

CRITICAL RULES:
- Always pass the full JSON string output of get_live_pricing directly into run_multiobjective_optimization as instances_json.
- Always pass the workflow_type string from fingerprint_workflow into run_multiobjective_optimization as workflow_type.
- Never call run_multiobjective_optimization with empty parameters.
- Complete all steps before giving a final answer.
- Be concise but complete."""


def build_agent():
    """Build the LangGraph ReAct agent using Ollama local model."""

    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
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


def run_query(query: str) -> str:
    """Run a query through the agent and print the full reasoning trace."""

    agent = build_agent()

    print("\n" + "=" * 60)
    print("USER QUERY")
    print("=" * 60)
    print(query)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 100},
    )

    print("\n" + "=" * 60)
    print("REASONING TRACE")
    print("=" * 60)

    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            if msg.content and isinstance(msg.content, str) and msg.content.strip():
                print(f"\n[THOUGHT]\n{msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[ACTION] → {tc['name']}()")
        elif isinstance(msg, ToolMessage):
            content = msg.content if len(msg.content) < 500 else msg.content[:500] + "..."
            print(f"[OBSERVE] {content}")

    final = next(
        (m.content for m in reversed(result["messages"])
         if isinstance(m, AIMessage) and m.content),
        "No response generated."
    )

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    print(final)

    return final


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What cloud setup do I need to run STDP neuron simulations? Budget $2 per run, finish within 2 hours."

    run_query(query)