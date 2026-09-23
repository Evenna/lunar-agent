"""
retriever.py — 离线中文知识库检索引擎（零第三方依赖，纯 Python 标准库）

设计目标：
  1. 完全离线、可在 RK3588 等资源受限板子上运行，不依赖 jieba/faiss/torch/numpy。
  2. 中文友好：内置基于知识库词表的正向最大匹配分词 + 二元组(bigram)回退，
     对术语型知识库（月球/宇航服/通信等专有名词）召回效果好。
  3. BM25 排序：经典、稳定、可解释，适合小规模精编知识库。

用法：
    kb = KnowledgeBase("knowledge_base")
    hits = kb.search("宇航员出舱前要检查什么", top_k=3)
    for score, entry in hits: ...
"""

import json
import math
import os
import re
from collections import defaultdict


# ---------- 中文/英文混合分词 ----------

# 从知识库中动态构建的词表会用于最大匹配；此处是一批高频领域词，
# 保证即使某词在正文里被拆开，也能作为整体被切出，提升检索精度。
SEED_TERMS = [
    "月球", "月面", "月尘", "月壤", "月海", "月夜", "月球车", "月球基地", "宇航服", "宇航员",
    "生命保障系统", "生命保障", "气闸舱", "永久阴影区", "二氧化碳", "太阳能", "电解水",
    "萨巴捷反应", "水冰", "水循环", "中继卫星", "鹊桥", "玉兔号", "玉兔二号", "阿波罗",
    "通信延迟", "无线电", "电磁波", "深空测控", "潮汐锁定", "月球背面", "微陨石", "辐射",
    "真空", "气压", "热控", "液冷服", "头盔", "面罩", "手套", "出舱", "PLSS", "MR", "VR",
    "低重力", "惯性", "陨石坑", "永久阴影", "储能", "燃料电池", "废弃物", "资源循环",
    "心理健康", "共餐", "圆桌", "植物", "作息", "睡眠", "运动", "人机协作", "机器人",
    "职业", "基地主管", "领航员", "联络员", "工程师", "教官", "比邻", "深空食栖",
    "星载净域", "月面启舷", "月球驾驶舱", "月球接待大厅", "来自未知的信号", "波形声墙",
    "情感", "拥抱", "击掌", "挥手", "昼夜", "温差", "朔望月", "重力", "氧气", "湿度",
    "太空", "航天", "空间站", "地球", "出舱活动", "月球探索", "月面基地", "餐食",
    "吃饭", "用餐", "共餐", "社群", "陪伴", "孤独", "隔离", "沉浸式", "展览",
]

_EN_NUM = re.compile(r"[a-zA-Z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")

# 通用停用字/词：这些字词区分度低，作为查询词会引入噪声，检索时过滤掉。
STOPWORDS = set(
    "的 了 是 在 有 和 与 吗 呢 啊 吧 呀 么 什 么 怎 样 谁 哪 里 为 什么 怎么 "
    "这 那 个 们 我 你 他 她 它 会 能 要 请 问 一 下 还 也 都 就 对 把 被 让 "
    "多 少 大 小 高 低 上 下 中 里 外 时 候 时候 一个 什么样 干 嘛 干嘛 介绍 "
    "关于 一些 可以 需要 如何 为何 到底 这个 那个".split()
)


def _build_vocab(*text_blobs):
    """把 SEED_TERMS 和知识库里出现的 2~6 字中文串合并成一个最大匹配词表。"""
    vocab = set(SEED_TERMS)
    return vocab


def tokenize(text, vocab):
    """
    中文：正向最大匹配（优先切出词表里的长词），未命中的连续中文串
          回退成 单字 + 相邻二元组（bigram），兼顾召回与精度。
    英文/数字：整体作为一个 token（小写）。
    """
    tokens = []
    i, n = 0, len(text)
    max_len = 6
    while i < n:
        ch = text[i]
        if _CJK.match(ch):
            # 尝试最大匹配词表
            matched = None
            for L in range(min(max_len, n - i), 1, -1):
                cand = text[i:i + L]
                if cand in vocab:
                    matched = cand
                    break
            if matched:
                tokens.append(matched)
                i += len(matched)
            else:
                # 单字 + 与下一个中文字组成 bigram
                tokens.append(ch)
                if i + 1 < n and _CJK.match(text[i + 1]):
                    tokens.append(text[i:i + 2])
                i += 1
        else:
            m = _EN_NUM.match(text, i)
            if m:
                tokens.append(m.group(0).lower())
                i = m.end()
            else:
                i += 1  # 跳过标点/空白
    return tokens


# ---------- BM25 ----------

class BM25:
    def __init__(self, corpus_tokens, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = corpus_tokens
        self.N = len(corpus_tokens)
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = []            # 每篇文档的词频
        self.df = defaultdict(int)
        for doc in corpus_tokens:
            freq = defaultdict(int)
            for t in doc:
                freq[t] += 1
            self.tf.append(freq)
            for t in freq:
                self.df[t] += 1
        self.idf = {}
        for t, df in self.df.items():
            # 带平滑的 BM25 idf
            self.idf[t] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query_tokens, index):
        freq = self.tf[index]
        dl = self.doc_len[index]
        s = 0.0
        for t in query_tokens:
            if t not in freq:
                continue
            idf = self.idf.get(t, 0.0)
            tf = freq[t]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += idf * (tf * (self.k1 + 1)) / denom
        return s


# ---------- 知识库封装 ----------

class KnowledgeBase:
    def __init__(self, kb_dir):
        self.entries = []       # [{topic, subtopic, question, answer, tags}, ...]
        self._load(kb_dir)
        self.vocab = _build_vocab()
        # 对每条目建索引文本：主题+子主题+问题+标签 权重更高（重复拼接），正文也计入
        corpus_tokens = []
        for e in self.entries:
            weighted = " ".join([
                e.get("topic", ""), e.get("subtopic", ""),
                e.get("question", ""), e.get("question", ""),      # 问题重复一次加权
                " ".join(e.get("tags", [])), " ".join(e.get("tags", [])),
                e.get("answer", ""),
            ])
            corpus_tokens.append(tokenize(weighted, self.vocab))
        self.bm25 = BM25(corpus_tokens)
        # 领域词集合：词表 + 所有条目的 tags + 主题/子主题词。
        # 查询必须至少命中其中一个领域词，才认为"和月球/太空相关"，否则直接拒答。
        # 这样能把"红烧肉/股票"挡在门外，同时放行"比邻/鹊桥/月球车"等短领域词。
        self.domain_terms = set(self.vocab)
        for e in self.entries:
            for t in e.get("tags", []):
                self.domain_terms.add(t)
            self.domain_terms.add(e.get("topic", ""))
            self.domain_terms.add(e.get("subtopic", ""))
        self.domain_terms.discard("")

    def _load(self, kb_dir):
        files = sorted(
            f for f in os.listdir(kb_dir) if f.endswith(".json")
        )
        for fn in files:
            path = os.path.join(kb_dir, fn)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            for e in data:
                self.entries.append(e)
        if not self.entries:
            raise RuntimeError(f"知识库为空，检查目录：{kb_dir}")

    def search(self, query, top_k=3, min_score=5.0, rel_ratio=0.25):
        """
        检索最相关条目。
        先判断查询是否命中任一"领域词"（月球/太空相关），否则直接拒答；
        再用 BM25 排序，min_score/rel_ratio 过滤弱相关噪声。
        """
        q_tokens = [t for t in tokenize(query, self.vocab) if t not in STOPWORDS]
        if not q_tokens:
            return []
        # 领域相关性闸门：查询里至少要有一个词属于知识库领域词，否则视为无关问题
        if not any(t in self.domain_terms for t in q_tokens):
            return []
        scored = []
        for idx in range(len(self.entries)):
            s = self.bm25.score(q_tokens, idx)
            if s > 0:
                scored.append((s, self.entries[idx]))
        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored or scored[0][0] < min_score:
            return []
        cutoff = scored[0][0] * rel_ratio
        scored = [pair for pair in scored if pair[0] >= cutoff]
        return scored[:top_k]

    def __len__(self):
        return len(self.entries)


if __name__ == "__main__":
    kb = KnowledgeBase(os.path.join(os.path.dirname(__file__), "knowledge_base"))
    print(f"知识库载入完成，共 {len(kb)} 条。")
    for q in ["宇航员出舱前要检查什么", "月球上能听到声音吗", "比邻机器人是干嘛的", "月球一天多长"]:
        print(f"\n【问】{q}")
        for score, e in kb.search(q, top_k=2):
            print(f"  [{score:.2f}] {e['topic']}/{e['subtopic']}: {e['question']}")
