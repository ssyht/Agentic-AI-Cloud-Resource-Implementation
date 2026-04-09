"""
agent.py — LangGraph ReAct Cloud Resource Configuration Agent
"""

import os
import sys
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
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

When a user asks about cloud setup for a scientific workflow, follow these steps IN ORDER:
1. Call fingerprint_workflow to classify the workflow
2. Call get_live_pricing with providers=["AWS","GCP","Azure"], exclude_gpu based on fingerprint
3. Call run_multiobjective_optimization with the pricing results
4. If the user mentioned a budget or time constraint, call constraint_filter
5. Call recommend_execution_plan for the best config
6. Give the user a clear final recommendation

Always complete all steps and give a full answer. Never stop early."""


def build_agent():
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
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
    agent = build_agent()

    print("\n" + "=" * 60)
    print("USER QUERY")
    print("=" * 60)
    print(query)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"recursion_limit": 100},
    )

    # Print the full reasoning trace
    print("\n" + "=" * 60)
    print("REASONING TRACE")
    print("=" * 60)
    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            pass  # already printed
        elif isinstance(msg, AIMessage):
            if msg.content and isinstance(msg.content, str) and msg.content.strip():
                print(f"\n[THOUGHT]\n{msg.content}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n[ACTION] → {tc['name']}()")
        elif isinstance(msg, ToolMessage):
            # Truncate long tool outputs for readability
            content = msg.content if len(msg.content) < 400 else msg.content[:400] + "..."
            print(f"[OBSERVE] {content}")

    # Final response is the last AIMessage
    final = next(
        (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
        "No response generated."
    )

    print("\n" + "=" * 60)
    print("HERE IS THE FINAL RECOMMENDATION")
    print("=" * 60)
    print(final)

    return final


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What cloud setup do I need to run STDP neuron simulations? Budget $2 per run, finish within 2 hours."

    run_query(query)
