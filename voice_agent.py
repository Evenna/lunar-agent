"""
voice_agent.py — 月球基地"语音问答模式"

这是给机器人用的"问答模式"入口：机器人平时做别的事，一旦被切换进本模式，
就进入【听观众说话 → 检索知识库 → 本地千问生成回答 → 语音播报】的循环，
直到听到退出词或被外部中止，再交还控制权给机器人主控。

它把三样东西串起来：
    voice_io.py  ——  耳朵(STT) + 嘴(TTS)
    agent.py     ——  大脑(LunarAgent：检索 + 千问生成)

两种进入方式：
  1) 命令行直接跑（开发/展台单机演示）：
        python3 voice_agent.py --role commander
  2) 被机器人主控当子进程/模块调用（见 run_qa_mode 函数和 ROBOT_INTEGRATION.md）。

设计要点（方便机器人对接）：
  - enter / exit 明确：进入时播欢迎语，退出词/信号可随时退出，退出播道别语。
  - 每一轮"听-答-播"都是独立的，机器人可以随时打断（kill 进程或发退出词）。
  - 语音全降级：没麦克风/没语音模型时自动退到键盘+打印，逻辑照样跑通。
  - 完全离线：STT / TTS / LLM 全部本地。
"""

import argparse
import sys

from agent import LunarAgent, ROLES
from voice_io import get_stt, get_tts


# 唤醒/退出词（机器人也可以用自己的唤醒机制，直接调 run_qa_mode 即可）
EXIT_WORDS = {"退出", "结束", "再见", "拜拜", "退出问答", "我不问了", "没有了"}
WELCOME = "你好，我是月球基地的讲解员，欢迎来到月球基地！有什么关于月球和太空的问题，尽管问我吧。"
GOODBYE = "好的，很高兴为你讲解，我们下次再聊，月球基地再见！"
NOT_HEARD = "抱歉我没有听清楚，能再说一遍吗？"


class VoiceQASession:
    """一次完整的语音问答会话（进入模式 → 多轮问答 → 退出模式）。"""

    def __init__(self, role="commander", backend="ollama", top_k=3,
                 stt_prefer="auto", tts_prefer="auto",
                 stt_kwargs=None, tts_kwargs=None):
        self.agent = LunarAgent(role=role, backend=backend, top_k=top_k)
        self.stt = get_stt(stt_prefer, **(stt_kwargs or {}))
        self.tts = get_tts(tts_prefer, **(tts_kwargs or {}))
        self.role_name = self.agent.role["name"]

    def say(self, text):
        """播报一句（终端也打印，方便现场/调试看）。"""
        print(f"\n【{self.role_name}】{text}\n", file=sys.stderr)
        self.tts.speak(text)

    def answer_once(self, question: str) -> str:
        """给一个文字问题，返回文字答案（不涉及语音，便于单元测试和 HTTP 复用）。"""
        reply, _hits = self.agent.answer(question)
        return reply

    def run(self, max_silence_rounds=3):
        """
        进入问答模式的主循环。听→答→播，直到退出词或连续听不清多次。
        机器人主控也可以不调这个，而是自己控制节奏、一轮一轮调 answer_once。
        """
        self.say(WELCOME)
        silence = 0
        while True:
            question = self.stt.listen_once().strip()

            if not question:
                silence += 1
                if silence >= max_silence_rounds:
                    self.say(GOODBYE)
                    break
                self.say(NOT_HEARD)
                continue
            silence = 0

            # 退出词
            if any(w in question for w in EXIT_WORDS):
                self.say(GOODBYE)
                break

            print(f"[听到] {question}", file=sys.stderr)
            reply = self.answer_once(question)
            self.say(reply)


def run_qa_mode(role="commander", backend="ollama", top_k=3,
                stt_prefer="auto", tts_prefer="auto"):
    """
    供机器人主控直接 import 调用的入口函数：
        from voice_agent import run_qa_mode
        run_qa_mode(role="commander")   # 切入语音问答模式，阻塞直到退出
    退出后函数返回，机器人主控即可收回控制权做别的事。
    """
    session = VoiceQASession(role=role, backend=backend, top_k=top_k,
                             stt_prefer=stt_prefer, tts_prefer=tts_prefer)
    session.run()


def main():
    ap = argparse.ArgumentParser(description="月球基地语音问答模式")
    ap.add_argument("--role", default="commander", choices=list(ROLES.keys()),
                    help="机器人角色人格")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "mock"],
                    help="推理后端：ollama=本机千问，mock=纯检索降级")
    ap.add_argument("--top_k", type=int, default=3, help="检索条数")
    ap.add_argument("--stt", default="auto", choices=["auto", "vosk", "whisper", "text"],
                    help="语音识别后端（auto 自动挑可用的，缺依赖降级到键盘输入）")
    ap.add_argument("--tts", default="auto", choices=["auto", "piper", "pyttsx3", "print"],
                    help="语音合成后端（auto 自动挑可用的，缺依赖降级到纯打印）")
    args = ap.parse_args()

    print(f"[语音问答模式] 角色：{ROLES[args.role]['name']} | 后端：{args.backend}",
          file=sys.stderr)
    run_qa_mode(role=args.role, backend=args.backend, top_k=args.top_k,
                stt_prefer=args.stt, tts_prefer=args.tts)


if __name__ == "__main__":
    main()
