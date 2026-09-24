# 月球基地问答 Agent（离线 RAG）

面向「未来月球基地」科普展览的**离线**问答程序。观众向机器人角色提问，Agent 先从本地知识库检索相关资料，交给本地大模型生成通俗、准确的口语化回答；知识库没覆盖的问题，模型也会用自身知识正常作答。**完全离线、不联网**。

## 它是什么

- **离线 RAG**：本地知识库（纯 Python BM25 检索，零第三方依赖）+ 本地大模型（Ollama 跑 `qwen2.5:1.5b`，CPU 即可）。检索到资料就依据资料答、保证事实准确；没检索到就让模型正常回答，不会甩「我不知道」。
- **7 个机器人角色**：不同展舱语气不同（见文末表）。
- **541 条知识库**：10 大主题，程序自动加载 `knowledge_base/` 下所有 `.json`。

## 安装

程序只用 Python 3 标准库，检索器零依赖。只需装 Ollama + 拉模型（一次即可，之后完全离线）：

```bash
curl -fsSL https://ollama.com/install.sh | sh   # 安装 ollama
ollama pull qwen2.5:1.5b                         # 拉模型（约 1GB）
```

## 使用

```bash
cd lunar-agent

# 单次提问
python3 agent.py --role navigator --ask "月球晚上那么长没太阳怎么发电？"

# 交互模式（不带 --ask 即进入循环，输入 q 退出）
python3 agent.py --role proxima

# 纯检索模式（不调模型，只看命中的资料，用于验证知识库或模型不可用时降级）
python3 agent.py --backend mock --ask "月球车的轮子和汽车一样吗？" --show_refs
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `--role` | 机器人角色（见文末表） | commander |
| `--backend` | `ollama`（本地模型）/ `mock`（纯检索降级） | ollama |
| `--ask` | 单次提问；不给则进入交互模式 | — |
| `--top_k` | 检索返回条数 | 3 |
| `--show_refs` | 打印命中的资料及分数 | 关 |

## 接入机器人（语音对接）

要让观众**用说的**提问、机器人**用语音**回答，提供两条路线，详见 **[`ROBOT_INTEGRATION.md`](ROBOT_INTEGRATION.md)**（给机器人老师的对接文档）：

- **路线 A（推荐）**：机器人管语音，Agent 只当「大脑」。起 HTTP 服务，机器人 POST 提问、拿文字答案自己念。
  ```bash
  python3 server.py --backend ollama --port 8080
  curl -s -X POST http://127.0.0.1:8080/ask -d '{"question":"月球上冷吗"}'
  ```
  接口：`POST /ask`、`GET /roles`、`GET /health`，纯标准库零依赖。
- **路线 B（兜底）**：Agent 把语音也全包（本地 STT + TTS，离线闭环）。
  ```bash
  python3 voice_agent.py --role commander
  ```
  没装语音模型时自动降级到键盘输入 + 文字打印，逻辑照样跑通。

## 扩充知识库

直接往 `knowledge_base/` 加 JSON 文件，程序启动自动加载。每条须含 `topic / subtopic / question / answer / tags` 五个字段。加完 `python3 retriever.py` 自检条数与检索结果。

## 机器人角色

| `--role` | 角色 | 展舱 | 语气 |
|----------|------|------|------|
| `commander` | 基地主管 | 接待大厅 | 亲切自信 |
| `eva` | EVA 装备教官 | 月面穿衣体验 | 可靠耐心、任务式 |
| `proxima` | 情感联络员 比邻 | 月球生活舱 | 温暖轻柔 |
| `navigator` | 月球交通领航员 | 月面行动舱 | 沉稳冷静、有指令感 |
| `steward` | 生活舱运营管理员 | 深空食栖 | 细致精确 |
| `signal` | 深空通信联络员 | 月球探索舱 | 冷静好奇 |
| `engineer` | 深空通讯工程师 | 烽火中央站 | 理性严谨 |

## 说明

- 机器人硬件（RK3588 等）本版不涉及，只在本机跑通问答逻辑；日后移植把 `llm_backend.py` 换成板子的推理后端即可，检索器和知识库原样复用。
- 命中知识库的回答严格依据检索资料、事实准确；库外问题由模型自身作答，个别措辞可能有小瑕疵。
