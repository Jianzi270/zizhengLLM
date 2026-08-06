"""GMM 软聚类模块（模块 B4）：对文档摘要向量做高斯混合模型软聚类。

流程：
  1. 读取 data/processed/summaries.jsonl（LLM 摘要）
  2. 用 BGE 嵌入模型对摘要向量化
  3. PCA 降维（保留 95% 方差，避免高维协方差估计不稳定）
  4. BIC 扫描选择最优类别数（--n-clusters 可手工指定）
  5. 拟合 GMM，输出软聚类概率（每篇文档属于各类的概率）
  6. 生成类别概览（类内文档数、代表性文档、级别/区域分布），供人工命名类别

用法：
  python -m src.cluster.gmm_cluster                    # 自动 BIC 选类别数
  python -m src.cluster.gmm_cluster --n-clusters 5     # 指定类别数

输出：
  data/processed/clusters.jsonl          # doc_id, cluster, conf, top_probs
  data/processed/cluster_overview.json   # 类别概览（代表性文档/区域级别分布/代表摘要）
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARIES_FILE = PROJECT_ROOT / "data" / "processed" / "summaries.jsonl"
METADATA_FILE = PROJECT_ROOT / "data" / "metadata" / "metadata.csv"
OUT_CLUSTERS = PROJECT_ROOT / "data" / "processed" / "clusters.jsonl"
OUT_OVERVIEW = PROJECT_ROOT / "data" / "processed" / "cluster_overview.json"


def load_docs():
    docs = []
    for line in SUMMARIES_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            docs.append(json.loads(line))
    return docs


def load_meta():
    import csv
    meta = {}
    if METADATA_FILE.exists():
        with METADATA_FILE.open(encoding="utf-8-sig", newline="") as fp:
            for row in csv.DictReader(fp):
                meta[row["filename"]] = row
    return meta


def select_k(vec_pca, max_k=15) -> int:
    from sklearn.mixture import GaussianMixture
    bics = {}
    for k in range(2, min(max_k + 1, vec_pca.shape[0] // 2)):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=3)
        gmm.fit(vec_pca)
        bics[k] = gmm.bic(vec_pca)
    best = min(bics, key=bics.get)
    print("BIC 扫描:", {k: round(v) for k, v in bics.items()})
    print(f"最优类别数（BIC）: {best}")
    return best


def main():
    parser = argparse.ArgumentParser(description="GMM 软聚类")
    parser.add_argument("--n-clusters", type=int, default=0, help="指定类别数（默认 BIC 自动选择）")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    docs = load_docs()
    print(f"加载摘要 {len(docs)} 篇")
    from src.embed.embed import embed_texts
    vec = embed_texts([d["summary"] for d in docs])
    print(f"摘要向量: {vec.shape}")

    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    pca = PCA(n_components=0.95, random_state=args.random_seed)
    vec_pca = pca.fit_transform(vec)
    print(f"PCA 降维: {vec.shape[1]} -> {vec_pca.shape[1]}（保留 95% 方差）")

    k = args.n_clusters or select_k(vec_pca)
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=args.random_seed, n_init=3)
    gmm.fit(vec_pca)
    prob = gmm.predict_proba(vec_pca)  # (n, k) 软概率
    labels = prob.argmax(axis=1)

    meta = load_meta()
    overview = []
    with OUT_CLUSTERS.open("w", encoding="utf-8", newline="\n") as fp:
        for doc, label, probs in zip(docs, labels, prob):
            top = {int(j): round(float(p), 3) for j, p in enumerate(probs) if p > 0.05}
            fp.write(json.dumps({"doc_id": doc["doc_id"], "cluster": int(label),
                                 "conf": round(float(probs[label]), 3), "top_probs": top},
                                ensure_ascii=False) + "\n")

    # 类别概览
    for c in range(k):
        idx = [i for i, l in enumerate(labels) if l == c]
        members = [docs[i] for i in idx]
        regions = Counter(meta.get(m["doc_id"], {}).get("region", "未知") for m in members)
        levels = Counter(meta.get(m["doc_id"], {}).get("source_level", "未知") for m in members)
        top_docs = sorted(idx, key=lambda i: prob[i][c], reverse=True)[:3]
        overview.append({
            "cluster": int(c),
            "size": len(idx),
            "top_region": regions.most_common(3),
            "top_level": levels.most_common(3),
            "top_docs": [docs[i]["doc_id"] for i in top_docs],
            "sample_summary": docs[top_docs[0]]["summary"][:100] if top_docs else "",
        })

    with OUT_OVERVIEW.open("w", encoding="utf-8", newline="\n") as fp:
        json.dump(overview, fp, ensure_ascii=False, indent=2)

    print(f"聚类完成：{len(docs)} 篇 -> {k} 类")
    for c in overview:
        print(f"  类{c['cluster']}: {c['size']} 篇 | 区域 {c['top_region']} | 级别 {c['top_level']}")
    print(f"输出: {OUT_CLUSTERS}")
    print(f"输出: {OUT_OVERVIEW}")


if __name__ == "__main__":
    main()
