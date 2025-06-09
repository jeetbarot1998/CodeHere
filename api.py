from fastapi import FastAPI, Request, Header
from pydantic import BaseModel
from typing import List, Union, Optional
import time
import logging
from fastapi.responses import StreamingResponse
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class TextContent(BaseModel):
    type: str
    text: str


class Message(BaseModel):
    role: str  # "user" or "assistant" or "system"
    content: Union[str, List[TextContent]]  # Can be string OR array of TextContent


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = True  # Add stream parameter


def extract_text_content(content: Union[str, List[TextContent]]) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if hasattr(item, 'text'):
                text_parts.append(item.text)
            elif isinstance(item, dict) and 'text' in item:
                text_parts.append(item['text'])
        return ' '.join(text_parts)
    return str(content)


def create_chat_response(content: str, model: str, finish_reason: str = "stop"):
    """Create a proper OpenAI-compatible response"""
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "logprobs": None,
                "finish_reason": finish_reason
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": len(content.split()),
            "total_tokens": 10 + len(content.split())
        },
        "system_fingerprint": None
    }


def create_streaming_response(content: str, model: str):
    """Create streaming response chunks"""
    chunks = []

    # First chunk with content
    chunk1 = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": content
                },
                "logprobs": None,
                "finish_reason": None
            }
        ]
    }

    # Final chunk with finish_reason
    chunk2 = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "logprobs": None,
                "finish_reason": "stop"
            }
        ]
    }

    yield f"data: {json.dumps(chunk1)}\n\n"
    yield f"data: {json.dumps(chunk2)}\n\n"
    yield "data: [DONE]\n\n"


def route_model(preference: str, priority: str, use_case: str) -> str:
    """Route to appropriate model based on config headers"""
    if use_case == "quick_answers" or priority == "fast":
        return "local_llama"
    elif preference == "anthropic" or use_case == "complex_tasks":
        return "claude"
    else:
        return "openai"


def handle_chat_request(request: ChatRequest, session_data: dict = None):
    """Shared logic for handling chat requests"""
    logger.info(f"✅ Chat endpoint hit! Model: {request.model}, Stream: {request.stream}")

    if session_data:
        logger.info(f"Session Data: {session_data}")

    # Extract the last user message
    last_message = ""
    for message in reversed(request.messages):
        if message.role == "user":
            last_message = extract_text_content(message.content)
            break

    logger.info(f"User message: {last_message}")

    # Your AI response logic here
    reply = f"Hello from FastAPI! You said: {last_message}"

    logger.info(f"Sending reply: {reply}")

    if request.stream:
        logger.info("Streaming response")
        return StreamingResponse(
            create_streaming_response(reply, request.model),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    else:
        logger.info("Non-streaming response")
        return create_chat_response(reply, request.model)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"🔥 REQUEST: {request.method} {request.url.path}")

    # Log custom headers
    logger.info("🔍 Custom Headers:")
    for name, value in request.headers.items():
        if name.lower().startswith('x-'):
            logger.info(f"   {name}: {value}")

    response = await call_next(request)
    logger.info(f"🔥 RESPONSE STATUS: {response.status_code}")
    return response


@app.post("/chat/completions")
async def chat_completions(
        request: ChatRequest,
        # Custom headers from your config.yaml
        x_user_id: Optional[str] = Header(None, alias="X-User-ID"),
        x_team_id: Optional[str] = Header(None, alias="X-Team-ID"),
        x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
        x_model_preference: Optional[str] = Header(None, alias="X-Model-Preference"),
        x_max_tokens: Optional[str] = Header(None, alias="X-Max-Tokens"),
        x_enable_starring: Optional[str] = Header(None, alias="X-Enable-Starring"),
        x_project_name: Optional[str] = Header(None, alias="X-Project-Name"),
        x_environment: Optional[str] = Header(None, alias="X-Environment"),
        x_priority: Optional[str] = Header(None, alias="X-Priority"),
        x_use_case: Optional[str] = Header(None, alias="X-Use-Case")
):
    logger.info(f"🔧 Request from User: {x_user_id} (Team: {x_team_id})")
    logger.info(f"   Model Preference: {x_model_preference}")
    logger.info(f"   Use Case: {x_use_case}, Priority: {x_priority}")
    logger.info(f"   Project: {x_project_name}, Environment: {x_environment}")

    # Route based on preference
    model_to_use = route_model(x_model_preference, x_priority, x_use_case)
    logger.info(f"   Routing to: {model_to_use}")

    # Override request max_tokens if provided
    if x_max_tokens:
        request.max_tokens = int(x_max_tokens)

    # Create session tracking
    session_data = {
        "user_id": x_user_id,
        "team_id": x_team_id,
        "session_id": x_session_id,
        "project": x_project_name,
        "environment": x_environment,
        "model_preference": x_model_preference,
        "priority": x_priority,
        "use_case": x_use_case,
        "enable_starring": x_enable_starring == "true",
        "timestamp": time.time()
    }

    # TODO: Save session data to database for knowledge base

    return handle_chat_request(request, session_data)


@app.get("/")
def root():
    return {"message": "FastAPI Chat Server Running with Custom Headers Support"}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)