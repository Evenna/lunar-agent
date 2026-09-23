"""
voice_io.py — 离线语音输入输出层（可插拔）

给月球基地问答 Agent 加"耳朵"和"嘴"，用于机器人现场语音问答（观众说话、机器人语音回答）。
本模块只负责语音↔文字的转换，问答逻辑仍由 agent.py 的 LunarAgent 完成。

设计原则：
  - 完全离线：STT / TTS 都优先选可离线运行、模型可预先下载到本机的方案。
  - 可插拔：STT、TTS 各自抽象成统一接口，缺哪个依赖就降级，不让整程序崩。
  - 对机器人友好：如果机器人自带语音（麦克风/扬声器/SDK），可完全不用本模块，
    直接走 server.py 的 HTTP 接口传文字即可（见 ROBOT_INTEGRATION.md 路线 A）。

============================================================
STT（语音识别，说话 → 文字）：默认用 Vosk（离线、中文模型约 40MB~1.3GB 可选）
    安装： pip install vosk sounddevice
    模型： 到 https://alphacephei.com/vosk/models 下载中文模型
           （vosk-model-small-cn-0.22 小巧适合嵌入式；vosk-model-cn-0.22 更准）
           解压后把路径通过 VOSK_MODEL_PATH 环境变量或构造参数传入。

TTS（语音合成，文字 → 说话）：默认用 piper（离线、音质好、树莓派/RK3588 可跑）
    安装： pip install piper-tts   （或用 piper 可执行文件）
    模型： 下载中文语音模型 zh_CN-huayan-medium.onnx 等，路径通过 PIPER_MODEL_PATH 传入。
    降级： 若 piper 不可用，回退到 pyttsx3（离线，音质一般）；再不行则只打印文字。
============================================================
"""

import os
import sys
import wave
import json
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod


# ========================= STT：语音识别 =========================

class STTBackend(ABC):
    @abstractmethod
    def listen_once(self, max_seconds: float = 8.0) -> str:
        """录一段音并返回识别出的文字（空字符串表示没听清）。"""
        ...


class VoskSTT(STTBackend):
    """
    Vosk 离线中文语音识别。麦克风实时采集，检测到一句话说完就返回。
    需要： pip install vosk sounddevice ；下载中文模型。
    """
    def __init__(self, model_path: str = None, samplerate: int = 16000):
        try:
            import vosk  # noqa
            import sounddevice  # noqa
        except ImportError as e:
            raise RuntimeError(
                "Vosk STT 需要 vosk 和 sounddevice：pip install vosk sounddevice"
            ) from e
        self.model_path = model_path or os.environ.get("VOSK_MODEL_PATH", "")
        if not self.model_path or not os.path.isdir(self.model_path):
            raise RuntimeError(
                "未找到 Vosk 中文模型目录。请下载模型并用 VOSK_MODEL_PATH 指定，"
                "参见 https://alphacephei.com/vosk/models"
            )
        import vosk
        self.samplerate = samplerate
        self._model = vosk.Model(self.model_path)
        self._KaldiRecognizer = vosk.KaldiRecognizer

    def listen_once(self, max_seconds: float = 8.0) -> str:
        import sounddevice as sd
        import queue
        rec = self._KaldiRecognizer(self._model, self.samplerate)
        q = queue.Queue()

        def _cb(indata, frames, time_, status):
            q.put(bytes(indata))

        collected = ""
        blocks = int(max_seconds * self.samplerate / 4000) + 1
        with sd.RawInputStream(samplerate=self.samplerate, blocksize=4000,
                               dtype="int16", channels=1, callback=_cb):
            for _ in range(blocks):
                data = q.get()
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    collected = res.get("text", "")
                    if collected.strip():
                        break
        if not collected.strip():
            res = json.loads(rec.FinalResult())
            collected = res.get("text", "")
        # Vosk 中文输出常带空格，去掉
        return collected.replace(" ", "").strip()


class WhisperCppSTT(STTBackend):
    """
    whisper.cpp 离线识别（若机器人主机上已装 whisper.cpp 可执行文件）。
    录音走 arecord/sounddevice，转写走 whisper.cpp。此处给出接口骨架，
    具体命令随部署环境的 whisper.cpp 构建方式而定。
    """
    def __init__(self, whisper_bin="whisper", model_path=None, samplerate=16000):
        self.whisper_bin = whisper_bin
        self.model_path = model_path or os.environ.get("WHISPER_MODEL_PATH", "")
        self.samplerate = samplerate
        if not shutil.which(whisper_bin):
            raise RuntimeError(f"未找到 whisper.cpp 可执行文件：{whisper_bin}")

    def listen_once(self, max_seconds: float = 8.0) -> str:
        # 用 sounddevice 录 max_seconds 秒到临时 wav，再交给 whisper.cpp
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError as e:
            raise RuntimeError("需要 sounddevice + numpy 录音") from e
        rec = sd.rec(int(max_seconds * self.samplerate),
                     samplerate=self.samplerate, channels=1, dtype="int16")
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.samplerate)
            w.writeframes(rec.tobytes())
        try:
            cmd = [self.whisper_bin, "-m", self.model_path, "-l", "zh",
                   "-f", wav_path, "-otxt", "-nt"]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            text = (out.stdout or "").strip()
            return text
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass


class TextSTT(STTBackend):
    """
    降级/调试用：不接麦克风，从键盘读入文字，模拟"听到一句话"。
    没有麦克风或没装语音依赖时，整套语音流程仍能跑通做演示。
    """
    def listen_once(self, max_seconds: float = 8.0) -> str:
        try:
            return input("[语音降级·请打字模拟提问] 你：").strip()
        except (EOFError, KeyboardInterrupt):
            return ""


# ========================= TTS：语音合成 =========================

class TTSBackend(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """把文字读出来（播放到扬声器）。"""
        ...


class PiperTTS(TTSBackend):
    """
    piper 离线中文语音合成，音质好、可在树莓派/RK3588 上跑。
    需要 piper 可执行文件 + 中文 onnx 语音模型。
    合成成 wav 后用 aplay/afplay/系统播放器播放。
    """
    def __init__(self, model_path=None, piper_bin="piper", player=None):
        self.piper_bin = piper_bin
        self.model_path = model_path or os.environ.get("PIPER_MODEL_PATH", "")
        if not shutil.which(piper_bin):
            raise RuntimeError(f"未找到 piper 可执行文件：{piper_bin}")
        if not self.model_path or not os.path.isfile(self.model_path):
            raise RuntimeError("未找到 piper 中文语音模型（.onnx），用 PIPER_MODEL_PATH 指定")
        self.player = player or self._detect_player()

    @staticmethod
    def _detect_player():
        for p in ("aplay", "afplay", "paplay", "ffplay"):
            if shutil.which(p):
                return p
        return None

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            subprocess.run([self.piper_bin, "-m", self.model_path, "-f", wav_path],
                           input=text, text=True, capture_output=True, timeout=60)
            if self.player == "ffplay":
                subprocess.run(["ffplay", "-nodisp", "-autoexit", wav_path],
                               capture_output=True)
            elif self.player:
                subprocess.run([self.player, wav_path], capture_output=True)
            else:
                print(f"[TTS·无播放器] {text}")
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass


class Pyttsx3TTS(TTSBackend):
    """离线 TTS 降级方案，音质一般但无需下载模型。需要 pip install pyttsx3。"""
    def __init__(self, rate=170):
        try:
            import pyttsx3
        except ImportError as e:
            raise RuntimeError("pyttsx3 未安装：pip install pyttsx3") from e
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        self._engine.say(text)
        self._engine.runAndWait()


class PrintTTS(TTSBackend):
    """最终降级：不发声，只把回答打印出来，保证流程不崩、可演示。"""
    def speak(self, text: str) -> None:
        print(f"🔊 [语音输出] {text}")


# ========================= 工厂：自动挑可用的后端 =========================

def get_stt(prefer="auto", **kwargs) -> STTBackend:
    """
    prefer: 'vosk' / 'whisper' / 'text' / 'auto'
    auto：依次尝试 vosk → whisper → 降级到键盘输入（text）。
    """
    order = {"vosk": [VoskSTT], "whisper": [WhisperCppSTT], "text": [TextSTT]}.get(
        prefer, [VoskSTT, WhisperCppSTT, TextSTT])
    for cls in order:
        try:
            return cls(**{k: v for k, v in kwargs.items()
                          if k in cls.__init__.__code__.co_varnames})
        except Exception as e:
            print(f"[STT] {cls.__name__} 不可用：{e}", file=sys.stderr)
    print("[STT] 全部不可用，降级为键盘输入。", file=sys.stderr)
    return TextSTT()


def get_tts(prefer="auto", **kwargs) -> TTSBackend:
    """
    prefer: 'piper' / 'pyttsx3' / 'print' / 'auto'
    auto：依次尝试 piper → pyttsx3 → 降级到只打印（print）。
    """
    order = {"piper": [PiperTTS], "pyttsx3": [Pyttsx3TTS], "print": [PrintTTS]}.get(
        prefer, [PiperTTS, Pyttsx3TTS, PrintTTS])
    for cls in order:
        try:
            return cls(**{k: v for k, v in kwargs.items()
                          if k in cls.__init__.__code__.co_varnames})
        except Exception as e:
            print(f"[TTS] {cls.__name__} 不可用：{e}", file=sys.stderr)
    print("[TTS] 全部不可用，降级为纯文字打印。", file=sys.stderr)
    return PrintTTS()


if __name__ == "__main__":
    # 自检：不接麦克风也能跑（会降级到键盘+打印）
    print("== voice_io 自检 ==", file=sys.stderr)
    stt = get_stt("auto")
    tts = get_tts("auto")
    print(f"STT 后端：{type(stt).__name__} | TTS 后端：{type(tts).__name__}", file=sys.stderr)
    tts.speak("语音模块自检完成，你好，这里是月球基地。")
    q = stt.listen_once()
    print(f"识别到：{q}", file=sys.stderr)
    tts.speak(f"我听到你说：{q}")
