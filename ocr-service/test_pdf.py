import httpx
import base64
import time
import subprocess
import os
import signal
import pypdfium2 as pdfium
from PIL import Image
import io

def test_pdf_ocr():
    # 1. Start the server as a subprocess
    server_process = subprocess.Popen(
        ["uv", "run", "python", "main.py"],
        cwd=os.path.dirname(__file__),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print("Starting Robyn server for PDF test...")
    
    url = "http://127.0.0.1:8080/chat/completions"
    max_retries = 30
    ready = False
    
    # Wait for server to be ready
    for i in range(max_retries):
        try:
            with httpx.Client() as client:
                try:
                    client.post(url, json={})
                except httpx.ConnectError:
                    raise
                except Exception:
                    # If it's anything but ConnectError, the server is likely listening
                    pass
            ready = True
            print("Server is ready.")
            break
        except httpx.ConnectError:
            time.sleep(2)
            print(f"Waiting for server... ({i+1}/{max_retries})")
    
    if not ready:
        server_process.kill()
        print("Server failed to start in time.")
        return

    try:
        pdf_path = os.path.join(os.path.dirname(__file__), "..", "paper.pdf")
        print(f"Loading PDF: {pdf_path}")
        pdf = pdfium.PdfDocument(pdf_path)
        
        # Test first page only to save time
        page = pdf[0]
        # render to a PIL image
        bitmap = page.render(scale=2.0)
        pil_image = bitmap.to_pil()
        
        # convert to base64 jpeg
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
        
        print("Sending OCR request for first page of paper.pdf...")
        start_time = time.time()
        try:
            with httpx.Client(timeout=300.0) as client:
                response = client.post(url, json=request_body)
            
            duration = time.time() - start_time
            print(f"Response received in {duration:.2f}s")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("\n=== OCR RESULT ===")
                print(data["choices"][0]["message"]["content"])
                print("==================\n")
            else:
                print(f"Response Body: {response.text}")
            
        except Exception as e:
            print(f"Test failed with exception: {e}")
            out, err = server_process.communicate(timeout=5)
            print(f"Server STDOUT:\n{out}")
            print(f"Server STDERR:\n{err}")

    finally:
        # Cleanup
        print("Stopping server...")
        server_process.send_signal(signal.SIGINT)
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

if __name__ == "__main__":
    test_pdf_ocr()
