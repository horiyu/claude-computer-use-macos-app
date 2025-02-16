from flask import Flask, request, render_template_string, Response, stream_with_context
import asyncio
import os
import json
import base64
import queue
import threading

from computer_use_demo.loop import sampling_loop, APIProvider
from computer_use_demo.tools import ToolResult
from anthropic.types.beta import BetaMessage, BetaMessageParam
from anthropic import APIResponse

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        instruction = request.form.get("instruction", "").strip()
        if not instruction:
            return "入力が必要です", 400

        def generate():
            q = queue.Queue()

            def stream_callback(message):
                # action の内容だけをキューに積む
                q.put(message)

            def run_loop():

                asyncio.run(
                    run_sampling_loop(instruction, stream_callback=stream_callback)
                )
                q.put(None)

            t = threading.Thread(target=run_loop)
            t.start()

            while True:
                msg = q.get()
                if msg is None:
                    break
                else:
                    if isinstance(msg, list):
                        formatted = ""
                        for item in msg:
                            if item.get("type") == "text":
                              if formatted:
                                formatted += "<br>"
                              formatted += item.get("text", "").replace("\n", "<br>")
                            elif item.get("type") == "tool_use":
                              if formatted:
                                formatted += "<br>"
                              formatted += str(item.get("input", {}))
                        msg = formatted
                    else:
                        msg = str(msg).replace("\n", "<br>")
                yield f"<p>{msg}</p>\n"

        return Response(stream_with_context(generate()), mimetype="text/html")
    else:
        # チャット画面風HTML（入力欄固定, チャットログ表示エリア）
        return render_template_string(
            """
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="UTF-8">
            <title>Claude Computer Use Chat</title>
            <style>
              body { margin: 0; padding: 0; font-family: sans-serif; }
              #log { padding: 10px; margin-bottom: 80px; }
              #inputForm {
                position: fixed;
                bottom: 0;
                width: 100%;
                background: #eee;
                padding: 10px;
                box-sizing: border-box;
                display: flex;
              }
              #instruction { flex: 1; font-size: 16px; padding: 8px; }
              #sendButton { font-size: 16px; padding: 8px 16px; }
            </style>
          </head>
          <body>
            <div id="log"></div>
            <div id="inputForm">
              <input type="text" id="instruction" placeholder="例: Save an image of a cat to the desktop.">
              <button id="sendButton">送信</button>
            </div>
            <script>
              const logDiv = document.getElementById('log');
              const instructionInput = document.getElementById('instruction');
              const sendButton = document.getElementById('sendButton');

              function appendLog(html) {
                const p = document.createElement('p');
                p.innerHTML = html;
                logDiv.appendChild(p);
                window.scrollTo(0, document.body.scrollHeight);
              }

              function sendInstruction() {
                const instruction = instructionInput.value.trim();
                if (!instruction) return;
                // ユーザ入力をログに表示
                appendLog("<strong>You:</strong> " + instruction);
                instructionInput.value = "";
                fetch("/", {
                  method: "POST",
                  headers: { "Content-Type": "application/x-www-form-urlencoded" },
                  body: "instruction=" + encodeURIComponent(instruction)
                }).then(response => {
                  const reader = response.body.getReader();
                  const decoder = new TextDecoder();
                  function read() {
                    reader.read().then(({ done, value }) => {
                      if (done) return;
                      const chunk = decoder.decode(value);
                      appendLog(chunk);
                      read();
                    });
                  }
                  read();
                }).catch(err => {
                  console.error(err);
                  appendLog("エラーが発生しました。");
                });
              }
              
              sendButton.addEventListener('click', sendInstruction);
              instructionInput.addEventListener('keypress', function(e) {
                if (e.key === "Enter") {
                  e.preventDefault();
                  sendInstruction();
                }
              });
            </script>
          </body>
        </html>
        """
        )


async def run_sampling_loop(instruction: str, stream_callback=None):
    api_key = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
    if api_key == "YOUR_API_KEY_HERE":
        message = "環境変数 ANTHROPIC_API_KEY に API キーをセットしてください。"
        if stream_callback:
            stream_callback(message)
        return message

    provider = APIProvider.ANTHROPIC

    messages: list[BetaMessageParam] = [
        {
            "role": "user",
            "content": instruction,
        }
    ]

    output_collector = []

    def output_callback(content_block):
        if isinstance(content_block, dict) and content_block.get("type") == "text":
            # action の内容のみを送信（例: "左クリック", "〇〇と入力" など）
            action_text = content_block.get("text")
            output_collector.append(action_text)
            if stream_callback:
                stream_callback(action_text)

    def tool_output_callback(result: ToolResult, tool_use_id: str):
        # ツールからの出力は画面に表示させない
        if result.base64_image:
            os.makedirs("screenshots", exist_ok=True)
            filename = f"screenshots/screenshot_{tool_use_id}.png"
            with open(filename, "wb") as f:
                f.write(base64.b64decode(result.base64_image))

    def api_response_callback(response: APIResponse[BetaMessage]):
        try:
            data = json.loads(response.text)
            message = data.get("content", "")
            output_collector.append(message)
            if stream_callback:
                stream_callback(message)
        except Exception as e:
            message = "Error parsing API response: " + str(e)
            output_collector.append(message)
            if stream_callback:
                stream_callback(message)

    await sampling_loop(
        model="claude-3-5-sonnet-20241022",
        provider=provider,
        system_prompt_suffix="",
        messages=messages,
        output_callback=output_callback,
        tool_output_callback=tool_output_callback,
        api_response_callback=api_response_callback,
        api_key=api_key,
        only_n_most_recent_images=10,
        max_tokens=4096,
    )

    return "\n".join(str(x) for x in output_collector)


if __name__ == "__main__":
    import webbrowser

    port = 5000  # 必要に応じてポートを変更してください
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)  # デフォルトブラウザで URL を自動オープン
    app.run(host="127.0.0.1", port=port, debug=True)
