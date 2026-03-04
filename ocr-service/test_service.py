import pytest
import json
import base64
from unittest.mock import MagicMock, patch
import sys
import os

# Adds current directory to sys.path to import main
sys.path.append(os.path.dirname(__file__))

from main import chat_completions

class MockRequest:
    def __init__(self, body):
        self.body = body

@pytest.mark.asyncio
async def test_chat_completions_parsing():
    # Mock request body with a simple base64 image (fake data)
    fake_image_b64 = base64.b64encode(b"fake image data").decode('utf-8')
    request_data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fake_image_b64}"}}
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.0
    }
    
    request = MockRequest(json.dumps(request_data).encode('utf-8'))

    # Mock MLX functions and global model/processor
    with patch('main.decode_image') as mock_decode, \
         patch('main.apply_chat_template') as mock_apply, \
         patch('main.generate') as mock_generate, \
         patch('main.model') as mock_model, \
         patch('main.processor') as mock_processor, \
         patch('main.config') as mock_config:
        
        mock_decode.return_value = MagicMock() # Mock PIL Image
        mock_apply.return_value = "formatted prompt"
        mock_generate.return_value = MagicMock(text="Extracted OCR Text")
        
        response = await chat_completions(request)
        
        # Robyn's Response object uses 'description' for the body in some versions or 'body'
        # Since we use Response(..., description=...) in our implementation
        res_body = json.loads(response.description)
        
        assert res_body["choices"][0]["message"]["content"] == "Extracted OCR Text"
        assert res_body["model"] == "mlx-community/PaddleOCR-VL-1.5-4bit"
        assert mock_decode.called
        assert mock_apply.called
        assert mock_generate.called

@pytest.mark.asyncio
async def test_error_handling():
    # Invalid JSON
    request = MockRequest(b"invalid json")
    
    response = await chat_completions(request)
    assert response.status_code == 500
    res_body = json.loads(response.description)
    assert "error" in res_body
