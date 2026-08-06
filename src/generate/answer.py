"""C4 生成流程：检索结果（文档元数据+文本块）作为上下文，用 LLM 生成结构化政务答案。

完整链路：question → C1 LLM 增强 → C2 DC-RAG 三级检索 → 上下文拼接 → LLM 生成答案

用法：
  python -m src.generate.answer "深圳2025年GDP增长目标是多少？"   # 单条问答
  python -m src.generate.answer --eval                           # 用评测集批量生成（质量评测样例）

输出：
  --eval 模式生成 data/eval/generated_answers.jsonl（question/answer/sources/耗时）
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.query_pipeline import enhance_query, dc_rag_retrieve  # noqa: E402
from src.summarize.summarize import get_llm_settings, load_env  # noqa: E402

ANSWER_PROMPT = (
    "你是政府智库报告撰写助手。请根据下列参考资料回答用户问题。\n"
    "要求：\n"
    "1. 基于参考资料，使用专业、规范、准确的政务语言回答；\n"
    "2. 引用资料来源（在相关表述后标注来源文档名称）；\n"
    "3. 结构清晰，必要时分点陈述；\n"
    "4. 若参考资料无法覆盖问题，请明确说明并给出可获取该信息的建议；\n"
    "5. 直接输出回答正文，不要额外解释。\n\n"
    "参考资料：\n{context}\n\n"
    "用户问题：{question}\n"
)


def build_context(results: list[dict]) -> str:
    """将检索结果（文本块+元数据）拼接为参考资料上下文。"""
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[来源{i}] 文档：{r['doc_id']}\n内容：{r['text']}")
    return "\n\n".join(parts)


def generate_answer(question: str, top_c: int = 2, top_d: int = 3, top_k: int = 3) -> dict:
    """完整链路：增强 → 检索 → 生成。返回 {enhanced, sources, answer}。"""
    load_env()
    cfg = json.loads((PROJECT_ROOT / "src" / "summarize" / "config.json").read_text(encoding="utf-8"))
    key, base_url, model = get_llm_settings(cfg)
    if not key:
        raise RuntimeError("未配置 API Key（请检查 .env 中的 LLM_API_KEY）")

    enhanced = enhance_query(question)
    # 双路检索：增强查询 + 原查询（query 融合），合并去重后取 top-k，缓解年份偏移
    results = dc_rag_retrieve(enhanced, top_c, top_d, top_k)
    if question != enhanced:
        extra = dc_rag_retrieve(question, top_c, top_d, top_k)
        merged = {r["doc_id"]: r for r in extra}
        for r in results:
            if r["doc_id"] not in merged or r["score"] > merged[r["doc_id"]]["score"]:
                merged[r["doc_id"]] = r
        results = sorted(merged.values(), key=lambda x: -x["score"])[:top_k]
    if not results:
        return {"enhanced": enhanced, "sources": [], "answer": "未能从知识库检索到相关资料。"}
    context = build_context(results)

    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "你是专业的政务文本生成助手。"},
                {"role": "user", "content": ANSWER_PROMPT.format(context=context, question=question)},
            ],
            "temperature": 0.3,
            "max_tokens": 1500,
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"].get("content", "").strip()
    if not content:
        raise RuntimeError("LLM 返回内容为空")
    return {
        "enhanced": enhanced,
        "sources": [{"doc_id": r["doc_id"], "score": r["score"]} for r in results],
        "answer": content,
    }


def main():
    parser = argparse.ArgumentParser(description="C4 生成流程")
    parser.add_argument("question", nargs="?", help="用户问题")
    parser.add_argument("--eval", action="store_true", help="用评测集批量生成")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.eval:
        qfile = PROJECT_ROOT / "data" / "eval" / "eval_questions.jsonl"
        questions = [json.loads(l) for l in qfile.read_text(encoding="utf-8").splitlines() if l.strip()]
        out_file = PROJECT_ROOT / "data" / "eval" / "generated_answers.jsonl"
        rows = []
        print(f"评测生成 {len(questions)} 题 ...")
        for i, q in enumerate(questions, 1):
            try:
                r = generate_answer(q["question"], top_k=args.top_k)
                rows.append({"question": q["question"], "gold_doc_id": q["gold_doc_id"],
                             "enhanced": r["enhanced"], "sources": r["sources"],
                             "answer": r["answer"], "answer_chars": len(r["answer"])})
            except Exception as e:
                rows.append({"question": q["question"], "gold_doc_id": q["gold_doc_id"],
                             "error": str(e), "answer": ""})
            print(f"  [{i}/{len(questions)}] 完成" if rows[-1].get("answer") else f"  [{i}/{len(questions)}] 失败")
            time.sleep(0.3)
        with out_file.open("w", encoding="utf-8", newline="\n") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok = sum(1 for r in rows if r.get("answer"))
        print(f"完成：{ok}/{len(rows)} 题生成成功，输出 {out_file}")
        return

    if not args.question:
        parser.print_help()
        return
    r = generate_answer(args.question, top_k=args.top_k)
    print("=" * 60)
    print(f"问题: {args.question}")
    print(f"增强后: {r['enhanced']}")
    print(f"来源: {[s['doc_id'] for s in r['sources']]}")
    print("=" * 60)
    print(r["answer"])


if __name__ == "__main__":
    main()
