"""文档摘要生成模块（模块 B3）：为每篇文档生成摘要，供 GMM 聚类使用。

两种模式：
  --method llm        调用 LLM（OpenAI 兼容接口）生成 150 字以内的生成式摘要（推荐）
  --method extractive 无 API 依赖的提取式摘要（按位置+关键词选句，兜底）

LLM 配置（优先级：.env 文件 > src/summarize/config.json）：
  - 在项目根目录创建 .env（参考 .env.example），填入：
      LLM_API_KEY=sk-xxx            # Moonshot/DeepSeek/OpenAI 等兼容服务的 Key
      LLM_BASE_URL=https://...      # OpenAI 兼容服务地址
      LLM_MODEL=xxx                 # 模型名
  - .env 已被 git 忽略，不会提交

用法：
  python -m src.summarize.summarize --method llm          # 生成式摘要（需配置 .env）
  python -m src.summarize.summarize --method extractive   # 提取式摘要（无外部依赖）
  python -m src.summarize.summarize --force               # 忽略已生成，全量重跑

输出：
  data/processed/summaries.jsonl（doc_id, summary），支持增量续跑（已生成文档跳过）
"""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"
CLEANED_DIR = PROJECT_ROOT / "data" / "processed" / "cleaned"
ENV_FILE = PROJECT_ROOT / ".env"

PROMPT_TEMPLATE = (
    "你是政务文档分析助手。请为以下政府工作报告生成一份摘要，要求：\n"
    "1. 150 字以内，直接输出摘要正文，不要任何前后缀或解释；\n"
    "2. 概括文档的核心内容：主要成就、政策方向与重点工作领域。\n"
    "文档标题：{title}\n"
    "文档内容（节选）：\n{text}\n"
)


def load_env():
    """加载项目根目录 .env（不覆盖已存在的环境变量）。"""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and not os.getenv(k):
            os.environ[k] = v


def load_config() -> dict:
    with CONFIG_FILE.open(encoding="utf-8") as fp:
        return json.load(fp)


def get_llm_settings(cfg: dict) -> tuple[str, str, str]:
    """返回 (api_key, base_url, model)，优先级 .env > config.json。"""
    load_env()
    key = os.getenv("LLM_API_KEY") or os.getenv("MOONSHOT_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    llm = cfg["llm"]
    base_url = os.getenv("LLM_BASE_URL") or llm["base_url"]
    model = os.getenv("LLM_MODEL") or llm["model"]
    return key, base_url, model


def call_llm(cfg: dict, title: str, text: str) -> str:
    """调用 OpenAI 兼容接口生成摘要。"""
    key, base_url, model = get_llm_settings(cfg)
    llm = cfg["llm"]
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是专业的政务文本分析助手。"},
            {"role": "user", "content": PROMPT_TEMPLATE.format(title=title, text=text[: llm["max_input_chars"]])},
        ],
        "temperature": 0.3,
        "max_tokens": 300,
    }
    resp = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=llm["timeout"],
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def extractive_summary(text: str, num_sentences: int = 5) -> str:
    """提取式摘要：优先文档前部句子，兼顾高频关键词句（无外部依赖兜底）。"""
    import re
    sentences = re.findall(r"[^。！？；\n]+[。！？；]?", text)
    if not sentences:
        return text[:200]
    # 高频词（去除常见停用词）
    stop = set("的了一是在不和有大这中人上为个国年市政府报告工作建设发展经济社会新要也等"
               "与及并对而或和通过进一步深入持续积极推动加快推进全面实施加强着力坚持完善提升").__iter__()
    words = Counter(re.findall(r"[\u4e00-\u9fa5]{2,4}", text))
    for w in list(words):
        if len(w) < 2 or all(ch in stop for ch in w):
            del words[w]
    top = {w for w, _ in words.most_common(10)}
    # 评分：位置越靠前权重越高，含高频词加分
    scored = []
    for i, s in enumerate(sentences):
        score = 1.0 / (i + 1) + 0.05 * sum(1 for w in top if w in s)
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return "".join(s for _, s in scored[:num_sentences])


def existing_summaries(out_file: Path) -> dict:
    if not out_file.exists():
        return {}
    return {json.loads(l)["doc_id"]: l for l in out_file.read_text(encoding="utf-8").splitlines() if l.strip()}


def main():
    parser = argparse.ArgumentParser(description="文档摘要生成")
    parser.add_argument("--method", choices=["llm", "extractive"], default="llm")
    parser.add_argument("--force", action="store_true", help="全量重跑（忽略已生成）")
    args = parser.parse_args()

    cfg = load_config()
    out_file = PROJECT_ROOT / cfg["output_file"]
    existing = {} if args.force else existing_summaries(out_file)
    docs = sorted(CLEANED_DIR.glob("*.txt"))
    results = dict(existing)

    failed = []
    for i, f in enumerate(docs, 1):
        doc_id = f.name
        if doc_id in results:
            continue
        text = f.read_text(encoding="utf-8")
        if args.method == "llm":
            key, _, _ = get_llm_settings(cfg)
            if not key:
                print("错误：未找到 API Key。请在项目根目录创建 .env（参考 .env.example）填写 LLM_API_KEY 后重试")
                print("提示：也可先使用 --method extractive 提取式摘要跑通流程")
                return
            try:
                summary = call_llm(cfg, Path(doc_id).stem, text)
            except requests.RequestException as e:
                failed.append((doc_id, str(e)))
                print(f"  [{i}/{len(docs)}] 失败 {doc_id}: {e}")
                continue
        else:
            summary = extractive_summary(text, cfg["extractive"]["num_sentences"])
        results[doc_id] = json.dumps({"doc_id": doc_id, "summary": summary}, ensure_ascii=False)
        if i % 20 == 0 or i == len(docs):
            _save(out_file, results)
            print(f"  [{i}/{len(docs)}] 已生成 {len(results)} 篇摘要")
        time.sleep(0.3)  # 限速

    _save(out_file, results)
    print(f"完成：{len(results)}/{len(docs)} 篇摘要，失败 {len(failed)} 篇")
    for doc_id, err in failed[:5]:
        print(f"  - {doc_id}: {err[:80]}")
    print(f"输出: {out_file}")


def _save(out_file: Path, results: dict):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8", newline="\n") as fp:
        for doc_id in sorted(results):
            fp.write(results[doc_id] + "\n")


if __name__ == "__main__":
    main()
