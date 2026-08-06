"""资政大模型 — 自研智能体 Web 服务（替代 Dify 平台，无需 Dify API Key）。

架构（全部本地/自研，仅生成环节使用 .env 中的 LLM Key）：
  用户问题 → 安全合规检查（敏感词过滤）→ C1 LLM 输入增强 → C2 DC-RAG 三级检索
          → C4 LLM 生成结构化答案 → 输出合规检测 → 展示（含来源文档）→ 审计日志

启动：
  python app/server.py            # 访问 http://127.0.0.1:8000
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, render_template, request  # noqa: E402

from src.generate.answer import generate_answer  # noqa: E402
from src.security.compliance import audit_log, check_content  # noqa: E402

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    """问答接口：question -> {answer, sources, enhanced}（含输入/输出合规检测与审计）"""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "请输入问题"}), 400

    # 输入侧合规：命中敏感词直接拒绝（输出可控）
    in_hits = check_content(question)
    if in_hits:
        audit_log({"event": "blocked", "reason": "input_sensitive", "question": question, "hits": in_hits})
        return jsonify({"error": "输入包含不合规内容，已拒绝处理。"}), 400

    t0 = time.time()
    try:
        r = generate_answer(question, top_c=2, top_d=3, top_k=3)
        # 输出侧合规：命中敏感词则标记，不向用户返回（可审计）
        out_hits = check_content(r["answer"])
        if out_hits:
            audit_log({"event": "blocked", "reason": "output_sensitive", "question": question,
                       "hits": out_hits, "answer_excerpt": r["answer"][:200]})
            return jsonify({"error": "生成内容未通过合规检测，已拦截。请调整提问方式。"}), 500
        audit_log({"event": "ask", "question": question, "enhanced": r["enhanced"],
                   "sources": [s["doc_id"] for s in r["sources"]],
                   "cost_s": round(time.time() - t0, 2), "answer_chars": len(r["answer"])})
        return jsonify({
            "answer": r["answer"],
            "sources": r["sources"],
            "enhanced": r["enhanced"],
        })
    except Exception as e:  # 网络/Key 错误友好提示
        audit_log({"event": "error", "question": question, "error": str(e),
                   "cost_s": round(time.time() - t0, 2)})
        return jsonify({"error": f"生成失败：{e}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("资政大模型智能体已启动：http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False)
