"""数据清洗流水线（模块 A3）：可复现的文本规范化脚本。

用法：
  python scripts/clean_text.py                          # 清洗存量语料 -> data/processed/cleaned/
  python scripts/clean_text.py <文件或目录> [--output 目录]  # 清洗增量数据

清洗规则（每一步都保持幂等、不破坏正文语义）：
  1. 统一换行（\\r\\n -> \\n）并去除 BOM
  2. 去除每行首尾空白（含全角空格 \\u3000、TAB）
  3. 连续空行压缩为单个空行（保留段落分隔）
  4. 检测疑似页眉/页脚（全文重复出现的孤立短行），默认仅记录不删除

输出：
  <输出目录>/<同名>.txt             # 清洗后文本（UTF-8，段落以空行分隔）
  <输出目录>/../cleaning_report.csv # 处理报告（含疑似页眉列表，供人工复核）
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "gov_work_reports_sz_txt_cleaned"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "cleaned"

# 疑似页眉/页脚判定：孤立短行在全文出现次数阈值
HEADER_MIN_COUNT = 3
HEADER_MAX_LEN = 20
# 正文固定开头语（出现多次但属于正文，排除误报）
IGNORED_HEADERS = ("各位代表", "各位委员", "政府工作报告")


def decode_text(raw: bytes) -> tuple[str, str]:
    """解码：utf-8-sig（去 BOM），失败回退 gbk。返回 (文本, 编码名)。"""
    for enc in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return "", "解码失败"


def normalize(text: str) -> str:
    """核心规范化：统一换行、去行首尾空白（含全角空格）、压缩空行。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        # 去除首尾普通空格与全角空格，TAB 转空格后再去除
        line = line.replace("\t", " ").strip(" \u3000\u00a0")
        lines.append(line)
    # 压缩连续空行（2 个及以上 -> 1 个）
    out = []
    blank = False
    for line in lines:
        if line == "":
            if not blank:
                out.append("")
                blank = True
        else:
            out.append(line)
            blank = False
    # 去除开头/结尾多余空行
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def detect_headers(lines: list[str]) -> list[str]:
    """检测疑似页眉/页脚：全文重复出现的孤立短行（排除正文固定开头语）。"""
    counter = Counter(
        line for line in lines
        if 0 < len(line) <= HEADER_MAX_LEN and not line.startswith(IGNORED_HEADERS)
    )
    return [line for line, cnt in counter.items() if cnt >= HEADER_MIN_COUNT]


def detect_nav_residue(text: str) -> bool:
    """检测网页导航残留（政府信息公开网站抓取特征：含多个 '|' 分隔的栏目项）。"""
    for line in text.split("\n"):
        if line.count("|") >= 5 and len(line) < 500:
            return True
    return False


# 正文特征词（用于识别正文起点与"无正文"文件检测）
# 政府工作报告以"各位代表/现在，我代表"开场；法治政府建设报告以"现将……报告如下"开场
BODY_KEYWORDS = ("各位代表", "现在，我代表", "过去一年", "工作回顾", "请予审议", "报告如下")


def find_body_start(lines: list[str]) -> int:
    """识别正文起点：优先找含正文特征词的行，回退到第一个长段落行。

    返回起点行索引；无法识别返回 0（不删除任何内容）。
    """
    for i, line in enumerate(lines):
        if any(k in line for k in BODY_KEYWORDS):
            return i
    for i, line in enumerate(lines):
        if len(line) >= 100:
            return i
    return 0


def clean_file(src: Path, out: Path) -> dict:
    """清洗单个文件，返回处理记录。"""
    raw = src.read_bytes()
    text, encoding = decode_text(raw)
    if not text:
        return {"filename": src.name, "encoding": encoding, "input_chars": 0,
                "output_chars": 0, "removed_lines": 0, "removed_preamble": 0,
                "headers": "", "nav_residue": False, "suspicious_empty_body": True, "ok": False}
    lines_in = text.split("\n")
    normalized = normalize(text)
    lines_out = normalized.split("\n")
    # 压缩掉的空行数 = 输入空行数 - 输出空行数
    removed = (len(lines_in) - len([l for l in lines_in if l.strip()])) - \
              (len(lines_out) - len([l for l in lines_out if l.strip()]))
    # 删除正文起点之前的页眉/导航残留块
    body_start = find_body_start(lines_out)
    preamble_removed = 0
    if body_start > 0:
        preamble_removed = body_start
        lines_out = lines_out[body_start:]
    # 无正文检测：全文不含任何正文特征词 -> 疑似残缺文件（爬取未成功）
    suspicious_empty = not any(k in "\n".join(lines_out) for k in BODY_KEYWORDS)
    headers = detect_headers(lines_out)
    nav = detect_nav_residue("\n".join(lines_out))
    out.write_text("\n".join(lines_out) + "\n", encoding="utf-8", newline="\n")
    return {
        "filename": src.name,
        "encoding": encoding,
        "input_chars": len(text),
        "output_chars": len(normalized),
        "removed_lines": removed,
        "removed_preamble": preamble_removed,
        "headers": " | ".join(headers),
        "nav_residue": nav,
        "suspicious_empty_body": suspicious_empty,
        "ok": True,
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--output=")]
    out_dir = DEFAULT_OUTPUT
    if any(a.startswith("--output=") for a in sys.argv[1:]):
        out_dir = Path([a for a in sys.argv[1:] if a.startswith("--output=")][0].split("=", 1)[1])
    src = Path(args[0]) if args else DEFAULT_INPUT
    if not src.exists():
        print(f"错误：输入路径不存在：{src}")
        sys.exit(1)

    files = sorted(src.glob("*.txt")) if src.is_dir() else [src]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir.parent / "cleaning_report.csv"

    records = []
    for f in files:
        records.append(clean_file(f, out_dir / f.name))
    with report_file.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    ok = sum(1 for r in records if r["ok"])
    print(f"共 {len(records)} 个文件，清洗成功 {ok} 个")
    print(f"输出目录: {out_dir}")
    print(f"处理报告: {report_file}")
    with_headers = [r for r in records if r["headers"]]
    print(f"疑似页眉/页脚文件数: {len(with_headers)}（已记录，未删除，见报告）")
    for r in with_headers[:10]:
        print(f"  - {r['filename']}: {r['headers']}")
    nav_files = [r["filename"] for r in records if r["nav_residue"]]
    print(f"含网页导航残留文件数: {len(nav_files)}（已记录，未删除）")
    for name in nav_files[:10]:
        print(f"  - {name}")
    pre_files = [(r["filename"], r["removed_preamble"]) for r in records if r["removed_preamble"] > 0]
    print(f"删除前置页眉/导航块文件数: {len(pre_files)}")
    for name, n in pre_files[:10]:
        print(f"  - {name}: 删除 {n} 行")
    empty_files = [r["filename"] for r in records if r["suspicious_empty_body"] and r["ok"]]
    print(f"疑似无正文（残缺）文件数: {len(empty_files)}")
    for name in empty_files:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
