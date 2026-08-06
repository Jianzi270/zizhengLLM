"""C3 检索质量评估：对比 DC-RAG 三级检索与简化 RAG（扁平）的命中率。

指标：
  - Recall@K（文档级）：期望文档（gold_doc）是否出现在返回的前 K 个文档中
  - 分别统计 DC-RAG 与扁平检索的命中率，验证 DC-RAG 优于简化 RAG（NFR4）

评测集：data/eval/eval_questions.jsonl（12 个问题，覆盖国家/省/市/区与不同主题）

用法：
  python -m src.retrieval.evaluate [--top-c 2] [--top-d 3]

输出：
  data/eval/eval_results.jsonl（每题命中详情）+ 控制台汇总
"""
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "eval_questions.jsonl"
OUT_FILE = PROJECT_ROOT / "data" / "eval" / "eval_results.jsonl"


def dc_rag_top_docs(question_vec, kb, top_c, top_d) -> list[str]:
    """DC-RAG：类别 → 文档 两级筛选后的候选文档。"""
    idx = kb["index"]
    cat_scores = kb["category_vectors"] @ question_vec
    cat_hits = sorted(range(len(cat_scores)), key=lambda i: -cat_scores[i])[:top_c]
    cand_docs = set()
    for cid in cat_hits:
        cand_docs.update(idx["cluster_docs"][str(cid)])
    doc_scores = {d: float(kb["doc_vectors"][idx["doc_ids"].index(d)] @ question_vec) for d in cand_docs}
    return [d for d, _ in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_d]]


def flat_top_docs(question_vec, kb, top_d) -> list[str]:
    """简化 RAG：全库文档直接 top-k（无类别/层次筛选）。"""
    idx = kb["index"]
    scores = kb["doc_vectors"] @ question_vec
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_d]
    return [idx["doc_ids"][i] for i in order]


def dc_rag_chunks(question_vec, kb, top_c, top_d, top_k) -> list[dict]:
    """DC-RAG 块级：类别 → 文档 → 文本块 三级检索返回文本块。"""
    idx = kb["index"]
    cat_scores = kb["category_vectors"] @ question_vec
    cat_hits = sorted(range(len(cat_scores)), key=lambda i: -cat_scores[i])[:top_c]
    cand_docs = set()
    for cid in cat_hits:
        cand_docs.update(idx["cluster_docs"][str(cid)])
    doc_scores = {d: float(kb["doc_vectors"][idx["doc_ids"].index(d)] @ question_vec) for d in cand_docs}
    doc_hits = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_d]
    cand_chunks = [(i, c) for i, c in enumerate(idx["chunks"]) if c["doc_id"] in dict(doc_hits)]
    scores = sorted(((i, float(kb["chunk_vectors"][i] @ question_vec)) for i, _ in cand_chunks),
                    key=lambda x: x[1], reverse=True)[:top_k]
    return [idx["chunks"][i] for i, _ in scores]


def flat_chunks(question_vec, kb, top_k) -> list[dict]:
    """简化 RAG 块级：全库文本块直接 top-k。"""
    idx = kb["index"]
    scores = kb["chunk_vectors"] @ question_vec
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    return [idx["chunks"][i] for i in order]


def main():
    parser = argparse.ArgumentParser(description="检索质量评估")
    parser.add_argument("--top-c", type=int, default=2)
    parser.add_argument("--top-d", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3, help="块级返回文本块数")
    args = parser.parse_args()

    questions = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"评测问题数: {len(questions)}")

    from src.embed.embed import embed_queries
    from src.retrieval.kb import load_kb
    kb = load_kb()
    qvecs = embed_queries([q["question"] for q in questions])

    results = []
    n_dc = n_flat = 0
    mrr_dc = mrr_flat = 0.0
    n_dc_chunk = n_flat_chunk = 0
    for q, qv in zip(questions, qvecs):
        dc_docs = dc_rag_top_docs(qv, kb, args.top_c, args.top_d)
        flat_docs = flat_top_docs(qv, kb, args.top_d)
        dc_chunks = dc_rag_chunks(qv, kb, args.top_c, args.top_d, args.top_k)
        flat_chunks_ = flat_chunks(qv, kb, args.top_k)
        hit_dc = q["gold_doc_id"] in dc_docs
        hit_flat = q["gold_doc_id"] in flat_docs
        hit_dc_chunk = q["gold_doc_id"] in {c["doc_id"] for c in dc_chunks}
        hit_flat_chunk = q["gold_doc_id"] in {c["doc_id"] for c in flat_chunks_}
        n_dc += hit_dc
        n_flat += hit_flat
        n_dc_chunk += hit_dc_chunk
        n_flat_chunk += hit_flat_chunk
        # MRR@top_d：gold 在结果中的排序质量（1/rank）
        rank_dc = dc_docs.index(q["gold_doc_id"]) + 1 if q["gold_doc_id"] in dc_docs else 0
        rank_flat = flat_docs.index(q["gold_doc_id"]) + 1 if q["gold_doc_id"] in flat_docs else 0
        mrr_dc += 1.0 / rank_dc if rank_dc else 0.0
        mrr_flat += 1.0 / rank_flat if rank_flat else 0.0
        results.append({
            "question": q["question"],
            "gold_doc_id": q["gold_doc_id"],
            "dc_rag_hit": hit_dc,
            "dc_rag_rank": rank_dc,
            "flat_hit": hit_flat,
            "flat_rank": rank_flat,
            "dc_rag_chunk_hit": hit_dc_chunk,
            "flat_chunk_hit": hit_flat_chunk,
        })

    with OUT_FILE.open("w", encoding="utf-8", newline="\n") as fp:
        for r in results:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(questions)
    print(f"\n=== 文档级 Recall@{args.top_d}（top_c={args.top_c}）===")
    print(f"DC-RAG 命中: {n_dc}/{total}（{n_dc/total*100:.0f}%） MRR={mrr_dc/total:.3f}")
    print(f"简化RAG 命中: {n_flat}/{total}（{n_flat/total*100:.0f}%） MRR={mrr_flat/total:.3f}")
    print(f"\n=== 块级 Recall@{args.top_k}（返回 top-{args.top_k} 文本块所属文档含 gold）===")
    print(f"DC-RAG 命中: {n_dc_chunk}/{total}（{n_dc_chunk/total*100:.0f}%）")
    print(f"简化RAG 命中: {n_flat_chunk}/{total}（{n_flat_chunk/total*100:.0f}%）")
    diff = n_dc_chunk - n_flat_chunk
    if diff > 0:
        print(f"结论: DC-RAG 块级命中率优于简化 RAG（+{diff} 题）✓")
    elif diff == 0:
        print("结论: 块级命中率持平")
    else:
        print(f"结论: 简化 RAG 块级更优（-{diff} 题），需检查类别划分")
    for r in results:
        flag_dc = "✓" if r["dc_rag_hit"] else "✗"
        flag_flat = "✓" if r["flat_hit"] else "✗"
        print(f"  DC-RAG[{flag_dc}] 扁平[{flag_flat}]  {r['question'][:36]}")
    print(f"\n详情输出: {OUT_FILE}")


if __name__ == "__main__":
    main()
