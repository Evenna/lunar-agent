# 月球基地问答 Agent — 机器人对接说明

> 本文档面向**机器人主机部署方（做机器人的老师/工程师）**。
> 月球基地问答 Agent 已开发完成，本文说明如何把它接入机器人，让机器人在
> 展台现场进入"语音问答模式"，与观众语音一问一答。
>
> 联系人：内容/知识库负责人（张一文）。技术问题按本文分工沟通。

---

## 一、这是什么

一个**完全离线**的月球科普问答程序：

- 观众问一句关于月球/太空/基地生活的问题，它检索本地 500 条知识库，
  用**本机千问模型（Ollama qwen2.5:1.5b）**生成口语化回答。
- 支持 **7 个角色人格**（基地主管、EVA 教官、情感联络员……），语气不同、知识库共享。
- 不联网、不上传任何数据，全部本机运行。
- 事实准确、可溯源：回答严格基于知识库，问到范围外会礼貌说"不确定"，不瞎编。

**你（机器人方）要做的事，只有一句话：让机器人在需要时"切进"这个问答模式，
把观众的话变成文字问题喂给它，把它返回的文字答案念出来。**

---

## 二、两种对接方式，请二选一

### ✅ 路线 A（强烈推荐）：机器人管语音，Agent 只当"大脑"

适用：**机器人本身自带麦克风、扬声器和语音能力（语音识别 + 语音合成 SDK）**。
绝大多数服务机器人都有。这种方式职责最清晰、最稳、最好调。

```
观众说话
  → 机器人自己的语音识别(STT)转成文字
    → HTTP POST 把文字发给 Agent 服务
      → Agent 返回文字答案
        → 机器人自己的语音合成(TTS)念出来
```

**Agent 侧启动一个 HTTP 服务即可（见 server.py）：**

```bash
cd lunar-agent
python3 server.py --host 127.0.0.1 --port 8080 --backend ollama
```

**机器人侧对接接口（详见第四节）：**
- 每收到一句观众的话，识别成文字后，`POST http://127.0.0.1:8080/ask`
- 拿到返回 JSON 里的 `answer` 字段，交给机器人 TTS 念出来

> 这种方式下，你**完全不需要**用我们的语音模块，语音全用你们机器人自己的，
> 音色、音量、口型联动都归你们控制。我们只保证"问一句文字、答一句文字"。

---

### 🅱 路线 B（兜底）：Agent 把语音也全包了

适用：**机器人没有好用的语音 SDK，或希望语音识别/合成也跑在本程序里。**

我们提供了完整的本地语音闭环 `voice_agent.py`：观众说话 → 本地语音识别
→ 检索+千问生成 → 本地语音合成播报，全离线。

```bash
cd lunar-agent
python3 voice_agent.py --role commander      # 切入语音问答模式，阻塞直到退出
```

机器人主控想"切进问答模式"时，可以：
- 作为**子进程**启动上面这条命令，退出词说"再见/退出"后进程结束，控制权交还；
- 或在 Python 里 `from voice_agent import run_qa_mode; run_qa_mode(role="commander")`。

> ⚠️ 路线 B 需要在机器人主机上额外装语音模型（见第五节），麦克风/扬声器要能被
> 本程序访问。如果机器人语音硬件被系统 SDK 独占，路线 B 可能不好使——那就走路线 A。

---

## 三、"切换到问答模式"怎么实现

你提到机器人平时做别的事，需要能**切换进入**问答模式。两种做法对应上面两条路线：

| | 进入问答模式 | 退出问答模式 |
|---|---|---|
| **路线 A** | 机器人主控开始把观众语音转文字、循环调 `/ask`，并念出 `answer` | 机器人主控停止调用即可（Agent 服务一直在后台待命，不占用时几乎零开销） |
| **路线 B** | 启动 `voice_agent.py` 子进程 / 调 `run_qa_mode()` | 观众说"退出/再见"，或机器人 kill 掉子进程 |

推荐 **A**：Agent 服务开机自启、常驻后台，机器人想问就问、想停就停，切换零成本。

---

## 四、HTTP 接口规范（路线 A 用）

服务地址：`http://<机器人主机>:8080`（端口可改）

### POST /ask — 提问
请求：
```json
{ "question": "月球上为什么这么冷", "role": "commander", "top_k": 3 }
```
- `question` **必填**，观众问题文字
- `role` 选填，角色 key（见 /roles），默认 `commander`
- `top_k` 选填，检索条数，默认 3

返回：
```json
{
  "ok": true,
  "role": "commander",
  "role_name": "基地主管",
  "answer": "月球几乎没有空气……（这句就是给机器人念出来的）",
  "refs": [ {"topic":"月球环境","subtopic":"极端温差","score":8.3} ],
  "in_kb": true
}
```
- `answer`：**机器人念这个字段**
- `in_kb`：`false` 表示问题超出知识库，已礼貌拒答，机器人照念 `answer` 即可
- `refs`：命中的知识条目（调试/展示用，可忽略）

### GET /roles — 列出可切换的角色
```json
{ "ok": true, "roles": [ {"key":"commander","name":"基地主管"}, ... ] }
```

### GET /health — 健康检查（机器人开机自检可调）
```json
{ "ok": true, "kb_size": 500, "backend": "ollama" }
```

### 调用示例（任何语言都行，这里用 curl）
```bash
curl -X POST http://127.0.0.1:8080/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"月球上冷吗","role":"eva"}'
```

> 首次每个角色的第一次请求会稍慢（加载模型上下文），之后就快了。
> 单次问答的模型生成耗时取决于机器人主机算力，qwen2.5:1.5b 在中端设备上约 1~4 秒。

---

## 五、部署要求（机器人主机需要满足）

### 必须（两条路线都要）
1. **Python 3.8+**
2. **Ollama + 千问模型**（这是"大脑"，必须有）：
   ```bash
   # 一次性联网安装，装完可离线
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull qwen2.5:1.5b          # 约 986MB，本项目用的模型
   # ollama 服务会自动在 127.0.0.1:11434 监听
   ```
   > 你提到机器人主机上已经部署了千问模型 —— 如果就是通过 Ollama 跑的，
   > **确认模型名是不是 `qwen2.5:1.5b`**。如果是别的名字/别的规格（比如
   > qwen2.5:7b、qwen2:1.5b），告诉我们，改一行配置即可（`llm_backend.py`
   > 里的 `model=` 或 server 启动加 `--` 参数），不用重训不用换库。
   > 如果不是走 Ollama 而是别的推理框架（vLLM / llama.cpp / 厂商 SDK），
   > 也告诉我们接口形式，我们适配 `llm_backend.py` 里一个 `generate()` 方法即可。

3. **把整个 `lunar-agent/` 目录拷到机器人主机**（含 knowledge_base/ 500 条知识库）。
   代码纯 Python 标准库，HTTP 服务**无需安装任何第三方包**。

### 仅路线 B（语音跑在本程序里）额外需要
- 语音识别（STT）：`pip install vosk sounddevice` + 下载中文模型
  （[Vosk 模型](https://alphacephei.com/vosk/models)，small-cn 约 40MB 适合嵌入式），
  用环境变量 `VOSK_MODEL_PATH` 指定模型目录。
- 语音合成（TTS）：`piper` + 中文语音模型（.onnx），用 `PIPER_MODEL_PATH` 指定；
  或退而求其次 `pip install pyttsx3`。
- 麦克风、扬声器可被本程序访问（未被机器人系统独占）。
- 缺依赖不会崩：语音模块会自动降级（STT→键盘输入、TTS→纯文字打印），便于先跑通逻辑。

### 开机自启（路线 A 推荐）
建议把 HTTP 服务做成开机自启（systemd 或机器人自己的服务管理），例如 systemd：
```ini
# /etc/systemd/system/lunar-agent.service
[Unit]
Description=Lunar Base QA Agent
After=network.target

[Service]
WorkingDirectory=/opt/lunar-agent
ExecStart=/usr/bin/python3 server.py --host 127.0.0.1 --port 8080 --backend ollama
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## 六、职责分工（谁做什么）

| 事项 | 内容方（张一文，已完成） | 机器人方（你们） |
|---|---|---|
| 知识库 500 条 + 检索 | ✅ 已完成 | — |
| 千问模型问答逻辑 | ✅ 已完成 | — |
| 7 角色人格 | ✅ 已完成 | — |
| HTTP 接口 / 语音闭环程序 | ✅ 已完成 | — |
| Ollama + 千问模型装在机器人主机 | 提供模型名/规格要求 | ✅ 安装、确认模型名 |
| 把观众语音转成文字（路线 A） | — | ✅ 用机器人自己的 STT |
| 把答案文字念出来（路线 A） | — | ✅ 用机器人自己的 TTS |
| "切换进入/退出问答模式"的触发逻辑 | 提供入口（HTTP / run_qa_mode） | ✅ 接到机器人交互流程里 |
| 麦克风/扬声器硬件 | — | ✅ 机器人自带 |

---

## 七、需要机器人方确认/提供的信息（沟通清单）

请就以下问题给内容方答复，据此我们做最后适配：

1. **机器人主机上的千问模型**：是通过 Ollama 跑的吗？模型名具体是什么
   （`ollama list` 看一下）？规格多大？—— 决定我们改不改配置。
2. 如果不是 Ollama，是什么推理框架/SDK？调用方式（HTTP？本地库？）是怎样的？
3. **机器人有没有自己的语音识别和语音合成能力？**（决定走路线 A 还是 B）
   - 有 → 走 A，你们调我们的 `/ask` HTTP 接口即可，最省事。
   - 没有 → 走 B，我们的 `voice_agent.py` 包语音，但你们需在主机装语音模型、
     并确认麦克风/扬声器能被本程序访问。
4. 机器人主机的操作系统、CPU/内存/有无 NPU？（评估模型生成速度）
5. "切换到问答模式"用什么触发？（唤醒词？按钮？后台指令？）—— 我们配合出入口。
6. 展台现场网络情况？（本方案不需要网，但确认一下没有强制联网依赖冲突）

---

## 八、快速验证（拿到程序后先跑这几条确认能用）

```bash
# 1. 确认模型在
ollama list        # 应能看到 qwen2.5:1.5b（或你们主机上的千问）

# 2. 不接语音、不接模型，纯逻辑自测（秒出，验证知识库和检索）
python3 agent.py --backend mock --ask "月球上冷吗" --show_refs

# 3. 接本机千问，真实问答（验证模型链路）
python3 agent.py --backend ollama --ask "宇航服为什么是白色的"

# 4. 起 HTTP 服务并测接口（路线 A 验收）
python3 server.py --backend ollama --port 8080 &
curl -s http://127.0.0.1:8080/health
curl -s -X POST http://127.0.0.1:8080/ask -H 'Content-Type: application/json' \
     -d '{"question":"月球的一天有多长"}'

# 5. 语音闭环自测（路线 B，无语音模型会自动降级到键盘+打印，逻辑照跑）
python3 voice_agent.py --backend mock
```

---

## 九、文件清单

| 文件 | 作用 |
|---|---|
| `agent.py` | 问答核心（检索 + 角色人格 + 千问生成），命令行可直接用 |
| `retriever.py` | 离线检索（BM25 + 中文分词，纯标准库） |
| `llm_backend.py` | 模型后端（Ollama 千问 / mock 降级），**换模型改这里** |
| `server.py` | **路线 A**：HTTP 接口服务（纯标准库，零依赖） |
| `voice_io.py` | **路线 B**：离线语音识别/合成层（可插拔、自动降级） |
| `voice_agent.py` | **路线 B**：语音问答模式主循环（听→答→播） |
| `knowledge_base/` | 500 条月球科普知识库（36 个 JSON） |
| `README.md` | 项目总说明 |
| `ROBOT_INTEGRATION.md` | 本文档 |
