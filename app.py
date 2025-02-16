from flask import Flask, request, render_template_string
import asyncio
import os
import json
import base64

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
        # 非同期処理を実行して結果を取得
        result = asyncio.run(run_sampling_loop(instruction))
        return f"<html><body><h2>結果</h2><pre>{result}</pre><br><a href='/'>戻る</a></body></html>"
    else:
        return render_template_string(
            """
        <html>
            <head><title>Claude Computer Use</title></head>
            <body>
                <h1>命令を入力してください</h1>
                <form method="post">
                    <input type="text" name="instruction" placeholder="例: Save an image of a cat to the desktop." size="60"><br><br>
                    <input type="submit" value="送信">
                </form>
            </body>
        </html>
        """
        )


async def run_sampling_loop(instruction: str):
    api_key = os.getenv("ANTHROPIC_API_KEY", "YOUR_API_KEY_HERE")
    if api_key == "YOUR_API_KEY_HERE":
        return "環境変数 ANTHROPIC_API_KEY に API キーをセットしてください。"

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
            output_collector.append("Assistant: " + content_block.get("text"))

    def tool_output_callback(result: ToolResult, tool_use_id: str):
        if result.output:
            output_collector.append(f"> Tool Output [{tool_use_id}]: {result.output}")
        if result.error:
            output_collector.append(f"!!! Tool Error [{tool_use_id}]: {result.error}")
        if result.base64_image:
            os.makedirs("screenshots", exist_ok=True)
            filename = f"screenshots/screenshot_{tool_use_id}.png"
            with open(filename, "wb") as f:
                f.write(base64.b64decode(result.base64_image))
            output_collector.append(f"Took screenshot {filename}")

    def api_response_callback(response: APIResponse[BetaMessage]):
        try:
            content = json.loads(response.text)["content"]
            formatted = json.dumps(content, indent=4)
            output_collector.append(
                "\n---------------\nAPI Response:\n" + formatted + "\n"
            )
        except Exception as e:
            output_collector.append("Error parsing API response: " + str(e))

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

    return "\n".join(output_collector)


if __name__ == "__main__":
    import webbrowser

    port = 5000  # 必要に応じてポートを変更してください
    url = f"http://127.0.0.1:{port}"
    webbrowser.open(url)  # デフォルトブラウザで URL を自動オープン
    app.run(host="127.0.0.1", port=port, debug=True)
