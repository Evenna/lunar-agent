"""
server.py — 月球基地问答 Agent 的 HTTP 接口服务（给机器人主控对接用）

这是【对接路线 A（推荐）】：机器人负责语音（自己的麦克风做语音识别、自己的
扬声器做语音播报），只把"识别出的文字问题"用 HTTP 发给本服务，拿到"文字答案"
后自己念出来。你的 Agent 只当"大脑"，语音交给机器人，职责最清晰、最好对接。

特点：
  - 纯 Python 标准库实现（http.server），零第三方依赖，本机直接跑，无需装框架。
  - 完全离线，只监听本机；模型走本机 Ollama。
  - 启动即把 Agent 和知识库常驻内存，避免每次请求重新加载。
  - 接口简单：POST /ask 提问、GET /roles 列角色、GET /health 健康检查。

启动：
    python3 server.py                       # 默认 0.0.0.0:8080
    python3 server.py --host 127.0.0.1 --port 9000 --backend ollama
    python3 server.py --backend mock        # 无模型时用纯检索降级，接口照样通

--------------------------------------------------------------------
接口约定（机器人老师按这个对接）：

POST /ask
  请求体(JSON): {"question": "月球上冷吗", "role": "commander", "top_k": 3}
      - question 必填；role 选填(默认 commander)；top_k 选填(默认 3)
  返回(JSON):   {"ok": true,
                 "role": "commander", "role_name": "基地主管",
                 "answer": "……回答文字……",
                 "refs": [{"topic":"月球环境","subtopic":"极端温差","score":8.3}, ...],
                 "in_kb": true}
      - answer 是给机器人念出来的文字
      - in_kb=false 表示问题超出知识库范围（已礼貌拒答，机器人照念即可）

GET /roles   → {"ok": true, "roles": [{"key":"commander","name":"基地主管"}, ...]}
GET /health  → {"ok": true, "kb_size": 500, "backend": "ollama"}
--------------------------------------------------------------------
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent import LunarAgent, ROLES

# 每个角色一个常驻 Agent 实例（共享同一份知识库检索器也行，这里简单起见按需建）
_AGENTS = {}
_CFG = {"backend": "ollama", "top_k": 3}


def get_agent(role: str) -> LunarAgent:
    role = role if role in ROLES else "commander"
    key = (role, _CFG["backend"], _CFG["top_k"])
    if key not in _AGENTS:
        _AGENTS[key] = LunarAgent(role=role, backend=_CFG["backend"],
                                  top_k=_CFG["top_k"])
    return _AGENTS[key]


class Handler(BaseHTTPRequestHandler):
    server_version = "LunarAgent/1.0"

    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # 精简日志，打到 stderr
        sys.stderr.write("[server] " + (fmt % args) + "\n")

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            any_agent = get_agent("commander")
            self._send(200, {"ok": True, "kb_size": len(any_agent.kb),
                             "backend": _CFG["backend"]})
        elif self.path.rstrip("/") == "/roles":
            roles = [{"key": k, "name": v["name"]} for k, v in ROLES.items()]
            self._send(200, {"ok": True, "roles": roles})
        else:
            self._send(404, {"ok": False, "error": "未知路径，可用：/ask /roles /health"})

    def do_POST(self):
        if self.path.rstrip("/") != "/ask":
            self._send(404, {"ok": False, "error": "未知路径，POST 只支持 /ask"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
        except Exception as e:
            self._send(400, {"ok": False, "error": f"请求体不是合法 JSON：{e}"})
            return

        question = (data.get("question") or "").strip()
        if not question:
            self._send(400, {"ok": False, "error": "缺少 question 字段"})
            return
        role = data.get("role") or "commander"
        top_k = data.get("top_k")

        try:
            agent = get_agent(role)
            if top_k and isinstance(top_k, int) and top_k != agent.top_k:
                agent.top_k = top_k
            reply, hits = agent.answer(question)
            refs = [{"topic": e["topic"], "subtopic": e["subtopic"],
                     "score": round(float(s), 2)} for s, e in hits]
            self._send(200, {
                "ok": True,
                "role": role if role in ROLES else "commander",
                "role_name": agent.role["name"],
                "answer": reply,
                "refs": refs,
                "in_kb": bool(hits),
            })
        except Exception as e:
            self._send(500, {"ok": False, "error": f"内部错误：{e}"})


def main():
    ap = argparse.ArgumentParser(description="月球基地问答 Agent HTTP 服务")
    ap.add_argument("--host", default="0.0.0.0", help="监听地址（默认 0.0.0.0）")
    ap.add_argument("--port", type=int, default=8080, help="端口（默认 8080）")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "mock"],
                    help="推理后端：ollama=本机千问，mock=纯检索降级")
    ap.add_argument("--top_k", type=int, default=3, help="默认检索条数")
    args = ap.parse_args()

    _CFG["backend"] = args.backend
    _CFG["top_k"] = args.top_k

    # 预热：启动即加载知识库和 Agent，首个请求不卡
    warm = get_agent("commander")
    print(f"[server] 月球基地问答服务启动：http://{args.host}:{args.port}", file=sys.stderr)
    print(f"[server] 知识库 {len(warm.kb)} 条 | 后端 {args.backend} | "
          f"接口：POST /ask, GET /roles, GET /health", file=sys.stderr)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] 已停止。", file=sys.stderr)
        httpd.shutdown()


if __name__ == "__main__":
    main()
