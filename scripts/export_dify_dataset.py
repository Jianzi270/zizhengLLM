"""B6 知识库数据集导出：将层次化知识库导出为标准数据集备份文件。

两种格式：
  - Dify 可导入格式（--format dify）：JSONL，每行 {"title": 文档名, "content": 文档全文}，
    可在 Dify 数据集"导入数据 → 文本/JSONL"中上传；也可作为通用语料备份。
  - 检索可复用格式（--format chunks）：从 data/processed/chunks.jsonl 复制为备份。

用法：
  python scripts/export_dify_dataset.py                 # 导出 Dify JSONL 数据集
  python scripts/export_dify_dataset.py --format chunks # 导出文本块备份

输出：
  data/export/dify_dataset.jsonl（159 篇文档全文）
  data/export/chunks_backup.jsonl（15509 文本块）
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "export"


def export_dify():
    """按文档聚合文本块，导出 Dify 可导入的 JSONL（title/content）。"""
    docs = defaultdict(list)
    for line in CHUNKS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        docs[c["doc_id"]].append(c["text"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "dify_dataset.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as fp:
        for doc_id in sorted(docs):
            fp.write(json.dumps({"title": doc_id, "content": "\n".join(docs[doc_id])}, ensure_ascii=False) + "\n")
    print(f"导出 Dify 数据集: {len(docs)} 篇文档 -> {out}")


def export_chunks():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "chunks_backup.jsonl"
    out.write_bytes(CHUNKS_FILE.read_bytes())
    n = sum(1 for l in CHUNKS_FILE.read_text(encoding="utf-8").splitlines() if l.strip())
    print(f"导出文本块备份: {n} 块 -> {out}")


def main():
    parser = argparse.ArgumentParser(description="知识库数据集导出")
    parser.add_argument("--format", choices=["dify", "chunks"], default="dify")
    args = parser.parse_args()
    export_dify() if args.format == "dify" else export_chunks()


if __name__ == "__main__":
    main()
