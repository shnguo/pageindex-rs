from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
import pypdfium2 as pdfium
import time
import os

pdf_path = "../paper.pdf"
print(f"Loading PDF: {pdf_path}")
pdf = pdfium.PdfDocument(pdf_path)
page = pdf[0]
bitmap = page.render(scale=1.0)
img = bitmap.to_pil()

MODEL_PATH = "mlx-community/PaddleOCR-VL-1.5-4bit"
print("Loading model...")
model, processor = load(MODEL_PATH)

messages = [
    {"role": "user", "content": "OCR the text in this image."}
]

print("Applying template...")
formatted_prompt = apply_chat_template(
    processor,
    model.config,
    messages,
    num_images=1,
)

print("Starting generation...")
start = time.time()
output = generate(
    model,
    processor,
    formatted_prompt,
    image=[img],
    max_tokens=1000,
    temperature=0.0,
)
end = time.time()
print(f"Generated in {end-start:.2f}s")
print(output)
