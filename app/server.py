"""资政大模型 — 自研智能体 Web 服务（替代 Dify 平台，无需 Dify API Key）。

架构（全部本地/自研，仅生成环节使用 .env 中的 LLM Key）：
  用户问题 → C1 LLM 输入增强 → C2 DC-RAG 三级检索（本地 BGE 嵌入 + 本地知识库）
          → C4 LLM 生成结构化答案 → 展示（含来源文档）

启动：
  python app/server.py            # 访问 http://127.0.0.1:8000
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from src.generate.answer import generate_answer  # noqa: E402

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    """问答接口：question -> {answer, sources, enhanced}"""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "请输入问题"}), 400
    try:
        r = generate_answer(question, top_c=2, top_d=3, top_k=3)
        return jsonify({
            "answer": r["answer"],
            "sources": r["sources"],
            "enhanced": r["enhanced"],
        })
    except Exception as e:  # 网络/Key 错误友好提示
        return jsonify({"error": f"生成失败：{e}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("资政大模型智能体已启动：http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False)
