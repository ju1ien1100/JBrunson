"""
WebSocket server that receives PDF pages (text + image) and lets the model
read/process each page. Replace `process_page` with your Magenta / vision
pipeline calls.
"""

import asyncio
import base64
import json

import websockets

HOST = "localhost"
PORT = 8765

# Max WebSocket frame size (50 MB). Bounds peak memory per message; must match
# (or exceed) the client's send size in pdf_reader.py.
MAX_FRAME_SIZE = 50 * 1024 * 1024


def process_page(page: dict) -> dict:
    """
    Where the model reads the page information.

    Hook your pipeline here:
      - vision LLM: caption + mood + speaker detection from `image_bytes`
      - text: dialogue extraction from `page["text"]`
      - Magenta: generate music from the resulting mood prompt
    Returns a result dict sent back to the client.
    """
    image_bytes = base64.b64decode(page["image_png_b64"])
    text = page["text"]
    page_no = page["page_number"]

    # --- TODO: replace with real model calls ---
    # caption = vision_model.describe(image_bytes)
    # mood = mood_from_caption(caption)
    # music = magenta.generate(style=mood)
    # voices = tts.synthesize(dialogue_lines)
    # -------------------------------------------

    summary = (
        f"page {page_no}: {len(image_bytes)} bytes image, "
        f"{len(text)} chars text"
    )

    return {
        "type": "page_result",
        "page_number": page_no,
        "summary": summary,
        # "music_b64": ...,
        # "voiceover_b64": ...,
    }


async def handler(websocket):
    print("Client connected.")
    async for message in websocket:
        try:
            data = json.loads(message)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON from client, skipping message: {exc}")
            continue

        if data.get("type") == "done":
            print("Client finished sending.")
            break

        if data.get("type") == "page":
            try:
                result = process_page(data)
            except (KeyError, ValueError) as exc:
                # The client blocks on recv() after each page, so we must
                # always reply — otherwise a failed page deadlocks it.
                print(f"Error processing page, sending error reply: {exc}")
                result = {
                    "type": "page_error",
                    "page_number": data.get("page_number"),
                    "error": str(exc),
                }
            await websocket.send(json.dumps(result))


async def main():
    print(f"Model server listening on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT, max_size=MAX_FRAME_SIZE):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())