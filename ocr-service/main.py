from robyn import Robyn, Request, Response, jsonify
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
import mlx.core as mx
from PIL import Image
import io
import base64
import json
import os
import signal

app = Robyn(__file__)

# Global model and processor
model = None
processor = None
config = None

MODEL_PATH = "mlx-community/PaddleOCR-VL-1.5-4bit"

def load_model():
    global model, processor, config
    print(f"Loading model: {MODEL_PATH}")
    model, processor = load(MODEL_PATH)
    config = model.config
    print("Model loaded successfully.")

@app.startup_handler
def setup():
    load_model()

def decode_image(image_url):
    if image_url.startswith("data:image"):
        # data:image/jpeg;base64,...
        header, encoded = image_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_data))
    else:
        # Assume it's a URL or something else not supported for now in this simple version
        raise ValueError("Only base64 data URIs are supported for now")

@app.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        body = json.loads(request.body)
        messages = body.get("messages", [])
        
        images = []
        processed_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            if isinstance(content, str):
                processed_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        img_url = part.get("image_url", {}).get("url")
                        if img_url:
                            images.append(decode_image(img_url))
                
                # Joins text parts and appends
                processed_messages.append({"role": role, "content": " ".join(text_parts)})

        print(f"Applying chat template with {len(images)} images...", flush=True)
        # Apply chat template with padding fix logic (passing images)
        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            image=images
        )

        print(f"Starting MLX generation for {len(images)} images...", flush=True)
        # Generate text
        output = generate(
            model,
            processor,
            formatted_prompt,
            image=images,
            max_tokens=body.get("max_tokens", 1000),
            temperature=body.get("temperature", 0.0),
        )
        print("MLX generation complete.", flush=True)

        response_data = {
            "id": "chatcmpl-robyn",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL_PATH,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output.text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        return jsonify(response_data)

    except Exception as e:
        print(f"Error processing request: {e}")
        return Response(
            status_code=500,
            headers={"Content-Type": "application/json"},
            description=json.dumps({"error": str(e)})
        )
    finally:
        mx.metal.clear_cache()

if __name__ == "__main__":
    app.start(port=8080, host="0.0.0.0")
