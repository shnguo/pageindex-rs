import httpx
import base64
import time

url = "http://127.0.0.1:8080/chat/completions"

# Wait for server
while True:
    try:
        httpx.get("http://127.0.0.1:8080/")
        break
    except:
        time.sleep(1)

print("Server is ready.")

# Small payload
print("Sending small payload...")
resp = httpx.post(url, json={
    "messages": [
        {"role": "user", "content": "hello"}
    ]
})
print("Small payload status:", resp.status_code)

# Real payload
import pypdfium2 as pdfium
from PIL import Image
import io

pdf_path = "../paper.pdf"
pdf = pdfium.PdfDocument(pdf_path)
page = pdf[0]
bitmap = page.render(scale=1.0)
pil_image = bitmap.to_pil()

buffered = io.BytesIO()
pil_image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

request_body = {
    "model": "mlx-community/PaddleOCR-VL-1.5-4bit",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "OCR the text in this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
            ]
        }
    ],
    "max_tokens": 1000,
    "temperature": 0.0
}

print("Sending large payload...")
start_time = time.time()
resp = httpx.post(url, json=request_body, timeout=300.0)
print(f"Large payload status: {resp.status_code} in {time.time()-start_time:.2f}s")
