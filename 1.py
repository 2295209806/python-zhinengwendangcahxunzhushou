# -*- coding: utf-8 -*-
"""
本地文档智能审查与摘要 Agent（完整版）
基础功能：读取 -> 分词(jieba) -> 词频 -> 词云 + 报告
进阶功能(步骤4)：
  ① 蒙特卡洛模拟（随机抽样 + 概率分布预测）
  ② 对接大模型 API（自动生成审查意见/摘要，无key自动降级）
  ③ watchdog 自动监控文件夹（新文档自动触发分析）

运行方式：
  python 1.py                 # 只跑基础功能（出词云）
  python 1.py --montecarlo    # 基础 + 蒙特卡洛模拟
  python 1.py --summarize     # 基础 + 大模型摘要
  python 1.py --all           # 全部功能一起跑
  python 1.py --watch         # 监控模式（放新文件自动分析）
"""
import csv
import os
import sys
from collections import Counter

import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud

jieba.setLogLevel(30)  # 静默 jieba 词典加载日志，输出更干净

# 让 matplotlib 弹窗标题也能正常显示中文
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ----------------- 配置区 -----------------
FOLDER_PATH = "./data"              # 待分析文档所在文件夹
STOPWORDS_FILE = "./stopwords.txt"  # 停用词表
USER_DICT = "./user_dict.txt"       # 自定义专业词典
OUTPUT_DIR = "./output"             # 结果输出目录
TOP_N = 50                          # 词云高频词数量
SHOW_PLOT = True                    # 是否弹出词云窗口

# 大模型 API 配置（优先读环境变量，没有就走"本地降级"）
LLM_API_KEY = os.environ.get("LLM_API_KEY", "sk-857fbfcfb36f42b89b0d1556c511be77")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")


def find_chinese_font():
    """自动寻找本机可用的中文字体，避免词云中文变方框。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def read_text_safely(file_path):
    """读取文本，自动尝试多种编码；读不出来返回 None 并提示。"""
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            print(f"  [跳过] {os.path.basename(file_path)} 读取失败: {e}")
            return None
    print(f"  [跳过] {os.path.basename(file_path)} 编码无法识别")
    return None


def load_stopwords(path):
    """加载停用词表；文件不存在时退回默认基础停用词。"""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return {"的", "了", "和", "在", "是", "我", "你", "他",
            "我们", "他们", "为", "以", "等", "并", "对", "与"}


def analyze_documents(folder_path=FOLDER_PATH,
                      stopwords_file=STOPWORDS_FILE,
                      user_dict=USER_DICT,
                      output_dir=OUTPUT_DIR,
                      top_n=TOP_N,
                      show_plot=None):
    """基础分析：分词 + 词频 + 词云 + 报告。成功返回 (all_words, top_words, per_file_counter)。"""
    if not os.path.isdir(folder_path):
        print(f"[错误] 文档文件夹不存在: {folder_path}")
        return None
    os.makedirs(output_dir, exist_ok=True)

    if user_dict and os.path.exists(user_dict):
        jieba.load_userdict(user_dict)
        print(f"[信息] 已加载自定义词典: {user_dict}")
    stopwords = load_stopwords(stopwords_file)
    print(f"[信息] 停用词数量: {len(stopwords)}")

    all_words = []
    per_file_counter = {}

    files = [f for f in sorted(os.listdir(folder_path))
             if f.endswith((".txt", ".md"))]
    if not files:
        print(f"[错误] 文件夹 {folder_path} 下没有找到 .txt/.md 文件")
        return None

    print(f"[信息] 共发现 {len(files)} 个文档，开始处理...")
    for filename in files:
        text = read_text_safely(os.path.join(folder_path, filename))
        if not text:
            continue
        words = jieba.lcut(text)
        filtered = [w for w in words
                    if len(w) > 1 and w not in stopwords and not w.isspace()]
        all_words.extend(filtered)
        per_file_counter[filename] = Counter(filtered)
        print(f"  [完成] {filename}: 有效词 {len(filtered)} 个")

    if not all_words:
        print("[错误] 未提取到有效词汇")
        return None

    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)
    print(f"\n[信息] 全局 Top {top_n} 高频词（前 10 预览）:")
    for w, c in top_words[:10]:
        print(f"    {w}: {c}")

    font_path = find_chinese_font()
    wc = WordCloud(
        font_path=font_path,
        width=1000, height=800,
        background_color="white",
        max_words=top_n,
        collocations=False,
    )
    wc.generate_from_frequencies(dict(top_words))
    image_path = os.path.join(output_dir, "document_wordcloud.png")
    wc.to_file(image_path)
    print(f"\n[输出] 词云已保存: {image_path}")

    report_txt = os.path.join(output_dir, "keywords_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("本地文档智能审查 —— 高频词报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"分析文档数: {len(per_file_counter)}\n")
        f.write(f"有效总词数: {len(all_words)}\n\n")
        f.write(f"全局 Top {top_n} 高频词:\n")
        for w, c in top_words:
            f.write(f"  {w}\t{c}\n")
    print(f"[输出] 全局报告已保存: {report_txt}")

    report_csv = os.path.join(output_dir, "keywords_per_file.csv")
    with open(report_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "关键词", "词频"])
        for filename, counter in per_file_counter.items():
            for w, c in counter.most_common(20):
                writer.writerow([filename, w, c])
    print(f"[输出] 分文件关键词表已保存: {report_csv}")

    if show_plot is None:
        show_plot = SHOW_PLOT
    if show_plot:
        plt.figure(figsize=(10, 8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("本地文档高频词云分析")
        plt.tight_layout()
        plt.show()

    print("[完成] 基础分析结束")
    return all_words, top_words, per_file_counter


# ============ 进阶功能①：蒙特卡洛模拟 ============
def montecarlo_analysis(all_words, top_words, output_dir=OUTPUT_DIR,
                        sims=2000, sample_size=100):
    """蒙特卡洛模拟：基于词频概率分布随机抽样，估计主题词稳定性。

    思路：把词频归一化成概率分布 p(w)，有放回随机抽取 sample_size 个词，
    重复 sims 次，每次统计「核心主题词(前5)」在样本中出现的比例，
    得到该比例的分布 -> 计算均值/标准差/95%置信区间，并画直方图。
    """
    import numpy as np

    print("\n" + "=" * 40)
    print("进阶功能①：蒙特卡洛模拟（随机抽样 + 概率分布预测）")
    print("=" * 40)

    counter = Counter(all_words)
    words = list(counter.keys())
    probs = np.array(list(counter.values()), dtype=float)
    probs = probs / probs.sum()

    theme_words = [w for w, _ in top_words[:5]]
    print(f"[信息] 研究对象(核心主题词): {', '.join(theme_words)}")
    print(f"[信息] 每次抽样 {sample_size} 个词，重复 {sims} 次...")

    hit_ratios = []
    for i in range(sims):
        sample = np.random.choice(words, size=sample_size, replace=True, p=probs)
        hits = sum(1 for w in sample if w in theme_words)
        hit_ratios.append(hits / sample_size)

    ratios = np.array(hit_ratios)
    mean = ratios.mean()
    std = ratios.std()
    ci_low = np.percentile(ratios, 2.5)
    ci_high = np.percentile(ratios, 97.5)

    print(f"\n[结果] 主题词命中率均值: {mean:.4f} ({mean*100:.2f}%)")
    print(f"[结果] 标准差: {std:.4f}")
    print(f"[结果] 95% 置信区间: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"[解读] 随机抽样 {sims} 次后，每 {sample_size} 个词里大约有 "
          f"{mean*sample_size:.1f} 个核心主题词，说明主题词高度集中、文档主题明确。")

    # 画直方图
    plt.figure(figsize=(8, 5))
    plt.hist(ratios, bins=40, color="#4C72B0", edgecolor="white", alpha=0.85)
    plt.axvline(mean, color="red", linestyle="--", label=f"均值 {mean:.4f}")
    plt.axvline(ci_low, color="gray", linestyle=":", label=f"95% CI [{ci_low:.4f}, {ci_high:.4f}]")
    plt.axvline(ci_high, color="gray", linestyle=":")
    plt.xlabel("主题词命中率")
    plt.ylabel("频次")
    plt.title("蒙特卡洛模拟：主题词命中率分布")
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(output_dir, "montecarlo_distribution.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"[输出] 蒙特卡洛分布直方图已保存: {hist_path}")

    # 作为"补充上下文"追加进报告
    report_txt = os.path.join(output_dir, "keywords_report.txt")
    with open(report_txt, "a", encoding="utf-8") as f:
        f.write("\n\n【蒙特卡洛模拟（补充上下文）】\n")
        f.write(f"  核心主题词: {'、'.join(theme_words)}\n")
        f.write(f"  抽样次数: {sims} 次, 每次 {sample_size} 词\n")
        f.write(f"  主题词命中率均值: {mean:.4f}\n")
        f.write(f"  95% 置信区间: [{ci_low:.4f}, {ci_high:.4f}]\n")
    print("[输出] 蒙特卡洛结果已追加到关键词报告")


# ============ 进阶功能②：对接大模型 API ============
def build_prompt(top_words, per_file_counter):
    """把关键词、词频、核心文本片段组装成 Prompt。"""
    kw_lines = "\n".join([f"  {w}: {c} 次" for w, c in top_words])
    snippet_lines = []
    for name, counter in per_file_counter.items():
        top = ", ".join([w for w, _ in counter.most_common(5)])
        snippet_lines.append(f"  - {name}: {top}")
    snippets = "\n".join(snippet_lines)

    return (
        "你是一名专业的文档审查助手。以下是对一批本地文档做词频分析后的结果，"
        "请据此生成一份结构化的文档审查意见，包含：1)文档主题概述；2)核心关键词解读；"
        "3)可能存在的问题或建议。要求简洁、专业，200字以内。\n\n"
        f"【Top{len(top_words)}关键词及词频】\n{kw_lines}\n\n"
        f"【各文档核心词】\n{snippets}\n"
    )


def local_summary(top_words, per_file_counter):
    """无 API Key 时的本地规则降级摘要。"""
    top10 = [w for w, _ in top_words[:10]]
    head_word, head_count = top_words[0]
    lines = [
        "【本地规则摘要（未调用大模型）】",
        "",
        f"本批次共分析 {len(per_file_counter)} 篇文档。",
        f"核心主题词为：{'、'.join(top10)}。",
        f"其中出现频率最高的是「{head_word}」，共 {head_count} 次，",
        "反映这批文档主要围绕该主题展开。",
        "",
        "（提示：设置环境变量 LLM_API_KEY 后，将自动调用大模型生成更专业的审查意见。）",
    ]
    return "\n".join(lines)


def llm_summarize(top_words, per_file_counter, output_dir=OUTPUT_DIR):
    """对接大模型 API 生成审查意见；无 key 或调用失败自动降级为本地摘要。"""
    print("\n" + "=" * 40)
    print("进阶功能②：对接大模型 API（自动生成审查意见/摘要）")
    print("=" * 40)

    prompt = build_prompt(top_words, per_file_counter)
    summary_path = os.path.join(output_dir, "summary_report.txt")

    if not LLM_API_KEY:
        print("[提示] 未配置 LLM_API_KEY，使用本地规则摘要降级方案")
        summary = local_summary(top_words, per_file_counter)
    else:
        try:
            import requests
            print(f"[信息] 正在调用大模型 {LLM_MODEL} ...")
            resp = requests.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            summary = data["choices"][0]["message"]["content"]
            print("[信息] 大模型返回成功")
        except Exception as e:
            print(f"[提示] 大模型调用失败({e})，降级为本地规则摘要")
            summary = local_summary(top_words, per_file_counter)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("本地文档智能审查 —— 智能摘要报告\n")
        f.write("=" * 40 + "\n\n")
        f.write(summary + "\n")
    print(f"[输出] 摘要报告已保存: {summary_path}")
    print("\n" + summary)


# ============ 进阶功能③：watchdog 自动监控 ============
def run_watch():
    """监控 data 文件夹，新文档放入自动触发分析，实现无人值守。"""
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class NewFileHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            if event.src_path.lower().endswith((".txt", ".md")):
                print(f"\n[监控] 检测到新文档: {os.path.basename(event.src_path)}")
                analyze_documents(show_plot=False)

    print("=" * 40)
    print("进阶功能③：watchdog 自动监控（无人值守智能审查 Agent）")
    print("=" * 40)
    print(f"[监控] 正在监控文件夹: {os.path.abspath(FOLDER_PATH)}")
    print("[监控] 放入新的 .txt/.md 文件即会自动分析，按 Ctrl+C 退出")

    observer = Observer()
    observer.schedule(NewFileHandler(), FOLDER_PATH, recursive=False)
    observer.start()
    try:
        observer.join()
    except KeyboardInterrupt:
        observer.stop()
        print("\n[监控] 已停止")
    observer.join()


if __name__ == "__main__":
    args = sys.argv[1:]

    # 监控模式：一直运行，不放词云弹窗
    if "--watch" in args or "-w" in args:
        run_watch()
        sys.exit(0)

    # 其它模式：先跑基础分析，再按需叠加进阶功能
    do_mc = ("--montecarlo" in args) or ("--all" in args)
    do_llm = ("--summarize" in args) or ("--all" in args)
    # 有进阶功能时，词云不弹窗，让流程自动跑完
    popup = not (do_mc or do_llm)

    result = analyze_documents(show_plot=popup)
    if result is None:
        sys.exit(1)
    all_words, top_words, per_file_counter = result

    if do_mc:
        montecarlo_analysis(all_words, top_words)
    if do_llm:
        llm_summarize(top_words, per_file_counter)
