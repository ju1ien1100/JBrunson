"""
PDF reader -> WebSocket client.
Reads a PDF page-by-page, extracts text and rendered page images,
and streams each page to the WebSocket server for the model to process.
"""

import asyncio
import base64
import json
import sys

import fitz  # PyMuPDF
import websockets

WS_URI = "ws://localhost:8765"


def extract_pages(pdf_path: str):
    """Yield one dict per page: page number, text, and a PNG image (base64)."""
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]

        # Extract text (speech bubbles / prose)
        text = page.get_text("text").strip()

        # Render the page to a PNG image (the panel/illustration)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("ascii")

        yield {
            "type": "page",
            "page_number": page_index + 1,
            "total_pages": len(doc),
            "text": text,
            "image_png_b64": img_b64,
        }
    doc.close()


async def send_pdf(pdf_path: str):
    async with websockets.connect(WS_URI, max_size=None) as ws:
        for page in extract_pages(pdf_path):
            await ws.send(json.dumps(page))
            print(f"Sent page {page['page_number']}/{page['total_pages']} "
                  f"({len(page['text'])} chars)")

            # Wait for the model's response for this page
            response = await ws.recv()
            result = json.loads(response)
            print(f"  -> model: {result.get('summary', result)}")

        # Signal completion
        await ws.send(json.dumps({"type": "done"}))
        print("All pages sent.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pdf_reader.py <path_to_pdf>")
        sys.exit(1)
    asyncio.run(send_pdf(sys.argv[1]))