from flask import Flask, request, render_template, Response, stream_with_context
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
                    yield f"<p>{str(msg).replace('\n', '<br>')}</p>\n"

        return Response(stream_with_context(generate()), mimetype="text/html")
    else:
        # テンプレートファイルを利用
        return render_template("index.html")


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
            action_text = content_block.get("text")
            output_collector.append(action_text)
            if stream_callback:
                stream_callback(action_text)

    def tool_output_callback(result: ToolResult, tool_use_id: str):
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

    port = 5000
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)
    app.run(host="127.0.0.1", port=port, debug=True)
