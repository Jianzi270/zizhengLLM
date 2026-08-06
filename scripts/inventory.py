"""数据盘点脚本（模块 A1）：扫描清洗后语料，校验编码与完整性，输出数据清单。

用法：python scripts/inventory.py
输出：data/inventory/data_inventory.csv（UTF-8 with BOM，Excel 可直接打开）
字段：filename, size_bytes, chars, lines, encoding_ok, encoding_note, head
"""
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "gov_work_reports_sz_txt_cleaned"
OUT_DIR = PROJECT_ROOT / "data" / "inventory"
OUT_FILE = OUT_DIR / "data_inventory.csv"


def read_text(f: Path):
    """优先按 utf-8-sig 读取，失败回退 gbk，返回 (文本, 编码备注)。"""
    try:
        return f.read_text(encoding="utf-8-sig"), ""
    except UnicodeDecodeError:
        try:
            return f.read_text(encoding="gbk"), "gbk"
        except UnicodeDecodeError as e:
            return "", f"utf-8/gbk 均失败: {e}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*.txt"))
    rows = []
    for f in files:
        size_bytes = f.stat().st_size
        text, note = read_text(f)
        rows.append({
            "filename": f.name,
            "size_bytes": size_bytes,
            "chars": len(text),
            "lines": text.count("\n") + 1 if text else 0,
            "encoding_ok": note == "",
            "encoding_note": note,
            "head": text[:50].replace("\r", "").replace("\n", " "),
        })
    with OUT_FILE.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(1 for r in rows if r["encoding_ok"])
    empty = [r["filename"] for r in rows if r["chars"] == 0]
    print(f"共 {len(rows)} 个文件，可正常解码 {ok} 个")
    if empty:
        print("警告：以下文件内容为空：", empty)
    print(f"输出: {OUT_FILE}")


if __name__ == "__main__":
    main()
