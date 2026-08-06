"""层次化知识库检索接口（模块 B5/C2）：按"类别 → 文档 → 文本块"三级检索（DC-RAG）。

用法：
  python -m src.retrieval.kb "深圳2026年经济社会发展目标" [--top-c 2] [--top-d 3] [--top-k 3]
  from src.retrieval.kb import load_kb, retrieve

三级检索流程：
  1. 类别级：查询向量与类别代表向量（category_vectors）比对，选 top C 类别
  2. 文档级：在选中类别内与文档代表向量（doc_vectors）比对，选 top D 文档
  3. 文本块级：在选中文档内与文本块向量（chunk_vectors）比对，选 top K 块
"""
import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
KB_DIR = PROJECT_ROOT / "data" / "knowledge_base"

_kb = None


def load_kb() -> dict:
    """加载知识库（向量 + 三级索引 + 嵌入模型），懒加载。"""
    global _kb
    if _kb is None:
        _kb = {
            "chunk_vectors": np.load(KB_DIR / "chunk_vectors.npy"),
            "doc_vectors": np.load(KB_DIR / "doc_vectors.npy"),
            "category_vectors": np.load(KB_DIR / "category_vectors.npy"),
            "index": json.loads((KB_DIR / "index.json").read_text(encoding="utf-8")),
        }
    return _kb


def _topk(scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in idx]


def retrieve(query: str, top_c: int = 2, top_d: int = 3, top_k: int = 3) -> list[dict]:
    """三级检索：类别 → 文档 → 文本块，返回命中的文本块列表（含元数据与相似度）。"""
    from src.embed.embed import embed_texts
    kb = load_kb()
    q = embed_texts([query])[0]

    idx = kb["index"]
    cat_vecs, doc_vecs, chunk_vecs = kb["category_vectors"], kb["doc_vectors"], kb["chunk_vectors"]

    # 1. 类别级检索
    cat_scores = cat_vecs @ q
    cat_hits = _topk(cat_scores, top_c)

    # 2. 文档级检索：在选中类别内
    cand_docs = set()
    for cid, _ in cat_hits:
        cand_docs.update(idx["cluster_docs"][str(cid)])
    doc_scores = {d: float(doc_vecs[idx["doc_ids"].index(d)] @ q) for d in cand_docs}
    doc_hits = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_d]

    # 3. 文本块级检索：在选中文档内
    cand_chunks = [(i, c) for i, c in enumerate(idx["chunks"]) if c["doc_id"] in dict(doc_hits)]
    if not cand_chunks:
        return []
    chunk_scores = [(i, float(chunk_vecs[i] @ q)) for i, _ in cand_chunks]
    chunk_hits = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    doc_cluster = idx["doc_cluster"]
    for i, score in chunk_hits:
        c = idx["chunks"][i]
        results.append({
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "cluster": doc_cluster.get(c["doc_id"]),
            "score": round(score, 4),
            "text": c["text"],
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="层次化知识库检索（DC-RAG 三级）")
    parser.add_argument("query", help="查询问题")
    parser.add_argument("--top-c", type=int, default=2, help="候选类别数")
    parser.add_argument("--top-d", type=int, default=3, help="候选文档数")
    parser.add_argument("--top-k", type=int, default=3, help="返回文本块数")
    args = parser.parse_args()

    load_kb()
    print(f"查询: {args.query}")
    print(f"知识库: 类别={len(load_kb()['category_vectors'])}, "
          f"文档={len(load_kb()['doc_vectors'])}, 文本块={len(load_kb()['chunk_vectors'])}")
    print("=" * 60)
    for r in retrieve(args.query, args.top_c, args.top_d, args.top_k):
        print(f"[score={r['score']:.3f} | 类{r['cluster']} | {r['doc_id']}]")
        print(f"  {r['text'][:100]}...")
    print("=" * 60)


if __name__ == "__main__":
    main()
