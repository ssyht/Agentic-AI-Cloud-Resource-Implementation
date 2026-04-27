"""
server.py — Flask backend for the OnTimeRecommend Chat UI

Setup:
  pip3 install flask flask-cors
  export ANTHROPIC_API_KEY=sk-ant-...
  PYTHONPATH=. python3 server.py

Then open: http://localhost:5000
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

# Import agent
sys_path_dir = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path_dir)

from agent.agent import run_agent


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Optional structured inputs from frontend
    kwargs = {
        "query": query,
        "max_cost_per_run": data.get("max_cost_per_run"),
        "max_runtime_hrs": data.get("max_runtime_hrs"),
        "required_provider": data.get("required_provider"),
        "pipeline_type": data.get("pipeline_type"),
        "decompose": data.get("decompose", False),
        "suggest_profiling": data.get("suggest_profiling", False),
        "full_data_size_gb": data.get("full_data_size_gb"),
    }

    # Remove None values
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        result = run_agent(**kwargs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    from tools.tools import log_run_feedback
    try:
        result = log_run_feedback.invoke(data)
        return jsonify(json.loads(result))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback/summary", methods=["GET"])
def feedback_summary():
    wf = request.args.get("workflow_type", "all")
    from tools.tools import get_feedback_summary
    result = get_feedback_summary.invoke({"workflow_type": wf})
    return jsonify(json.loads(result))


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    backend = os.environ.get("AGENT_LLM_BACKEND", "anthropic")

    print("\n" + "=" * 50)
    print("OnTimeRecommend Cloud Agent Server")
    print("=" * 50)
    print(f"  LLM Backend : {backend}")
    print(f"  API Key     : {'set' if api_key else 'NOT SET — export ANTHROPIC_API_KEY'}")
    print(f"  URL         : http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(debug=True, port=5000)