"""
llm_backend.py — 可插拔的大模型推理接口

统一接口：  backend.generate(prompt: str) -> str

提供两种后端：
  - OllamaBackend：调用本机 Ollama 跑 qwen2.5:1.5b（千问），真正用检索到的
    资料生成自然口语回答。默认后端。离线运行（模型已拉到本地）。
  - MockBackend：不加载任何模型，直接把检索到的知识条目整理成回答，
    用于无模型环境演示或作为 Ollama 不可用时的降级方案。

切换后端：get_backend("ollama") / get_backend("mock")，或用 agent.py 的 --backend 参数。
"""

import json
import urllib.request
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """输入完整提示词，返回模型生成的回答文本。"""
        ...


class OllamaBackend(LLMBackend):
    """
    调用本机 Ollama HTTP API 运行千问模型。

    前置条件（本机一次性配置，之后离线可用）：
      1. 安装 ollama：curl -fsSL https://ollama.com/install.sh | sh
      2. 拉模型：ollama pull qwen2.5:1.5b
      3. ollama 服务会自动在 127.0.0.1:11434 监听。
    """
    def __init__(self, model="qwen2.5:1.5b",
                 host="http://127.0.0.1:11434",
                 temperature=0.3, num_predict=512, timeout=120):
        self.model = model
        self.url = host.rstrip("/") + "/api/generate"
        self.temperature = temperature
        self.num_predict = num_predict
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            return (out.get("response") or "").strip()
        except Exception as e:
            # 模型不可用时降级到检索回显，保证程序不崩
            return _MockBackend_singleton.generate(prompt) + \
                f"\n\n（提示：模型调用失败，已降级为纯检索模式。原因：{e}）"


class MockBackend(LLMBackend):
    """
    不加载任何模型，直接把检索到的知识条目整理成回答。
    用于无模型环境演示，或作为模型加载失败时的降级方案。
    """
    def generate(self, prompt: str) -> str:
        marker = "【参考资料】"
        end = "【问题】"
        if marker in prompt and end in prompt:
            body = prompt.rsplit(marker, 1)[1].split(end, 1)[0].strip()
            return "（纯检索模式，未接入模型；以下为知识库命中内容）\n" + body
        return "（纯检索模式）未检索到相关资料。"


_MockBackend_singleton = MockBackend()


def get_backend(name="ollama", **kwargs) -> LLMBackend:
    """工厂函数：
        name='ollama' → 本机千问模型生成（默认，推荐）
        name='mock'   → 纯检索回显，零依赖，不需要模型
    """
    name = (name or "ollama").lower()
    if name == "mock":
        return _MockBackend_singleton
    return OllamaBackend(**kwargs)
