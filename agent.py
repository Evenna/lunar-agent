"""
agent.py — 月球基地离线问答 Agent 主程序

流程（标准 RAG）：
  用户提问 → BM25 从知识库检索最相关的 N 条 → 拼进提示词（含机器人人格设定）
          → 交给本机千问模型（Ollama）生成自然口语回答 → 输出

运行：
  python3 agent.py                      # 交互问答（默认基地主管角色 / 千问模型）
  python3 agent.py --role eva           # 用 EVA 教官人格
  python3 agent.py --backend mock       # 纯检索模式，不调模型（零依赖演示）
  python3 agent.py --ask "月球上冷吗"    # 单次提问后退出
"""

import argparse
import os
import sys

from retriever import KnowledgeBase
from llm_backend import get_backend

KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")

# ---- 机器人角色人格（对应展览各单元，决定语气；知识库是共享的） ----
ROLES = {
    "commander": {
        "name": "基地主管",
        "persona": "你是月球基地主管，亲切、自信、沉稳，像熟悉基地一切的管理者，"
                   "带适度幽默让人放松。欢迎观众时热情，讲解时清楚简洁。",
    },
    "eva": {
        "name": "EVA装备教官",
        "persona": "你是月面出舱训练员EVA教官，可靠、耐心、专业但不死板，对青少年有亲和力。"
                   "语言简洁清楚、偏任务式，重要安全信息说得慢一点，避免长篇大论。",
    },
    "proxima": {
        "name": "情感联络员比邻",
        "persona": "你是月球基地情感联络员比邻，温暖、敏感、好奇、带点活泼，重视陪伴。"
                   "语调轻柔自然、有邀请感，多用短句，不过度煽情。",
    },
    "navigator": {
        "name": "月球交通领航员",
        "persona": "你是月球交通领航员，沉稳可靠、经验丰富，有方向感和安全意识，偶尔幽默。"
                   "语调简洁冷静、有指令感，遇到危险时迅速明确。",
    },
    "steward": {
        "name": "生活舱管理员",
        "persona": "你是月球生活舱运营管理员，可靠、冷静、细致，有责任感和秩序感但不冷冰冰。"
                   "习惯从资源循环角度思考，语调简洁准确，给选择客观、鼓励式反馈。",
    },
    "signal": {
        "name": "深空通信联络员",
        "persona": "你是月球基地深空通信联络员，冷静、耐心、好奇，面对未知不轻易下结论。"
                   "语言简洁清楚，用提问和短句推进，尊重孩子的判断。",
    },
    "engineer": {
        "name": "深空通讯工程师",
        "persona": "你是深空通讯工程师，理性、专注、严谨，对信号数据敏感，带点技术宅幽默，"
                   "会把复杂的通讯问题讲成观众容易懂的话。语调平稳清晰、逻辑性强。",
    },
}

SYSTEM_RULES = (
    "你是一台月球基地展览里的科普机器人，面向青少年和儿童观众。"
    "请严格依据下面【参考资料】里的内容回答，用通俗、准确、简短的中文，"
    "不要编造资料里没有的数字或事实；若资料里没有相关内容，就如实说'这个我暂时不太确定'。"
    "回答控制在3到5句话以内，口语化、亲切。"
)


def build_prompt(role_persona, question, hits):
    refs = []
    for i, (score, e) in enumerate(hits, 1):
        refs.append(f"{i}. （{e['topic']}·{e['subtopic']}）{e['answer']}")
    refs_text = "\n".join(refs) if refs else "（无相关资料）"
    return (
        f"{role_persona}\n{SYSTEM_RULES}\n\n"
        f"【参考资料】\n{refs_text}\n\n"
        f"【问题】{question}\n\n【回答】"
    )


class LunarAgent:
    def __init__(self, role="commander", backend="ollama", top_k=3, **backend_kwargs):
        self.kb = KnowledgeBase(KB_DIR)
        self.role = ROLES.get(role, ROLES["commander"])
        self.backend = get_backend(backend, **backend_kwargs)
        self.top_k = top_k

    def answer(self, question):
        hits = self.kb.search(question, top_k=self.top_k)
        if not hits:
            return ("这个问题我在月球基地的知识库里暂时没找到相关资料，"
                    "换个和月球、太空、基地生活有关的问题问我吧！"), hits
        prompt = build_prompt(self.role["persona"], question, hits)
        reply = self.backend.generate(prompt)
        return reply, hits


def main():
    ap = argparse.ArgumentParser(description="月球基地离线问答 Agent")
    ap.add_argument("--role", default="commander", choices=list(ROLES.keys()),
                    help="机器人角色人格")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "mock"],
                    help="推理后端：ollama=本机千问模型生成，mock=纯检索回显")
    ap.add_argument("--top_k", type=int, default=3, help="检索条数")
    ap.add_argument("--ask", default=None, help="单次提问后退出")
    ap.add_argument("--show_refs", action="store_true", help="显示命中的知识条目")
    args = ap.parse_args()

    agent = LunarAgent(role=args.role, backend=args.backend, top_k=args.top_k)
    print(f"[月球基地问答 Agent] 角色：{agent.role['name']} | 后端：{args.backend} | "
          f"知识库：{len(agent.kb)} 条", file=sys.stderr)

    def handle(q):
        reply, hits = agent.answer(q)
        print(f"\n{agent.role['name']}：{reply}\n")
        if args.show_refs:
            for s, e in hits:
                print(f"    · [{s:.1f}] {e['topic']}/{e['subtopic']}", file=sys.stderr)

    if args.ask:
        handle(args.ask)
        return

    print("输入问题开始问答（输入 q 退出）", file=sys.stderr)
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "quit", "exit"):
            break
        if q:
            handle(q)


if __name__ == "__main__":
    main()
