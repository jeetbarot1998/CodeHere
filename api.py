from fastapi import FastAPI, Request, Header, Depends
from pydantic import BaseModel
from typing import List, Union, Optional, AsyncGenerator
import time
import logging
from fastapi.responses import StreamingResponse
import json
import asyncio
from datetime import datetime

# LangChain imports
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangChain AI Proxy Server", version="1.0.0")


class TextContent(BaseModel):
    type: str
    text: str


class Message(BaseModel):
    role: str  # "user" or "assistant" or "system"
    content: Union[str, List[TextContent]]


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = True


class CustomHeaders(BaseModel):
    user_id: Optional[str] = None
    team_id: Optional[str] = None
    session_id: Optional[str] = None
    model_preference: Optional[str] = None
    max_tokens: Optional[str] = None
    enable_starring: Optional[str] = None
    project_name: Optional[str] = None
    environment: Optional[str] = None
    priority: Optional[str] = None
    use_case: Optional[str] = None


class ResponseData(BaseModel):
    """Model for saving responses to database"""
    user_id: str
    team_id: str
    session_id: str
    project_name: str
    model_used: str
    messages: List[Message]
    response: str
    timestamp: datetime
    starred: bool = False
    metadata: dict = {}


# Model configuration
MODEL_CONFIGS = {
    "gemma": {
        "type": "ollama",
        "model_name": "gemma3:4B",
        "base_url": "http://localhost:11434"
    },
    "gpt-4": {
        "type": "openai",
        "model_name": "gpt-4",
        "api_key": "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"  # Set via environment variable
    },
    "claude": {
        "type": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "api_key": "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"  # Set via environment variable
    }
}


def get_langchain_model(model_name: str, temperature: float = 0.7, max_tokens: int = 2048):
    """Factory function to create LangChain model instances"""
    config = MODEL_CONFIGS.get(model_name.lower())

    if not config:
        # Default to Gemma for unknown models
        config = MODEL_CONFIGS["gemma"]

    if config["type"] == "ollama":
        return Ollama(
            model=config["model_name"],
            base_url=config.get("base_url", "http://localhost:11434"),
            temperature=temperature
        )
    elif config["type"] == "openai":
        return ChatOpenAI(
            model=config["model_name"],
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=config.get("api_key")  # Should be set via environment
        )
    elif config["type"] == "anthropic":
        return ChatAnthropic(
            model=config["model_name"],
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=config.get("api_key")  # Should be set via environment
        )
    else:
        raise ValueError(f"Unsupported model type: {config['type']}")


def convert_messages_to_langchain(messages: List[Message]) -> List:
    """Convert OpenAI format messages to LangChain format"""
    langchain_messages = []

    for message in messages:
        content = extract_text_content(message.content)

        if message.role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif message.role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        elif message.role == "system":
            langchain_messages.append(SystemMessage(content=content))

    return langchain_messages


def extract_text_content(content: Union[str, List[TextContent]]) -> str:
    """Extract text from content (same as before)"""
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


async def save_response_to_db(response_data: ResponseData):
    """Save response to database (placeholder for now)"""
    # TODO: Implement actual database saving
    logger.info(f"💾 Saving response to DB: {response_data.session_id}")
    logger.info(f"   Model: {response_data.model_used}")
    logger.info(f"   Response length: {len(response_data.response)} chars")

    # For now, just log. Later integrate with your DB
    pass


def route_model_by_preference(headers: CustomHeaders, request_model: str) -> str:
    """Route to appropriate model based on headers and request"""
    # Priority order: explicit model request > use case > preference > default

    if request_model and request_model.lower() in MODEL_CONFIGS:
        return request_model.lower()

    if headers.use_case == "quick_answers" or headers.priority == "fast":
        return "gemma"  # Fast local model
    elif headers.model_preference == "anthropic":
        return "claude"
    elif headers.model_preference == "openai":
        return "gpt-4"
    else:
        return "gemma"  # Default to local model


async def stream_langchain_response(
        model,
        messages: List,
        model_name: str,
        response_data: ResponseData
) -> AsyncGenerator[str, None]:
    """Stream response from LangChain model"""
    try:

        # For Ollama models (like Gemma), use invoke with streaming
        if isinstance(model, Ollama):
            # Ollama doesn't support streaming callbacks in the same way
            # So we'll get the full response and simulate streaming
            response = await asyncio.to_thread(
                lambda: model.invoke(messages)
            )

            # Simulate streaming by yielding chunks
            words = response.split()
            full_response = ""

            for i, word in enumerate(words):
                chunk_content = word + " " if i < len(words) - 1 else word
                full_response += chunk_content

                chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": chunk_content
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Save the complete response
            response_data.response = full_response.strip()
            await save_response_to_db(response_data)

        else:
            # For OpenAI/Anthropic models with proper streaming
            async for chunk in model.astream(messages):
                chunk_data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": chunk.content
                            },
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

        # Final chunk
        final_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "api_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"


async def get_langchain_response(
        model,
        messages: List,
        model_name: str,
        response_data: ResponseData
) -> dict:
    """Get non-streaming response from LangChain model"""
    try:
        response = await asyncio.to_thread(
            lambda: model.invoke(messages)
        )

        response_text = response if isinstance(response, str) else str(response)

        # Save response
        response_data.response = response_text
        await save_response_to_db(response_data)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,  # Placeholder
                "completion_tokens": len(response_text.split()),
                "total_tokens": 10 + len(response_text.split())
            }
        }
    except Exception as e:
        logger.error(f"Error getting response: {e}")
        raise


async def get_custom_headers(
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
) -> CustomHeaders:
    return CustomHeaders(
        user_id=x_user_id,
        team_id=x_team_id,
        session_id=x_session_id,
        model_preference=x_model_preference,
        max_tokens=x_max_tokens,
        enable_starring=x_enable_starring,
        project_name=x_project_name,
        environment=x_environment,
        priority=x_priority,
        use_case=x_use_case
    )


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
        headers: CustomHeaders = Depends(get_custom_headers)
):
    logger.info(f"🔧 LangChain Request from User: {headers.user_id}")
    logger.info(f"   Model Preference: {headers.model_preference}")
    logger.info(f"   Use Case: {headers.use_case}, Priority: {headers.priority}")

    # Route to appropriate model
    model_to_use = route_model_by_preference(headers, request.model)
    logger.info(f"   🎯 Routing to: {model_to_use}")

    # Override max_tokens if provided
    max_tokens = request.max_tokens
    if headers.max_tokens:
        max_tokens = int(headers.max_tokens)

    # Get LangChain model instance
    try:
        model = get_langchain_model(model_to_use, request.temperature, max_tokens)
        logger.info(f"   ✅ Model loaded: {type(model).__name__}")
    except Exception as e:
        logger.error(f"   ❌ Failed to load model: {e}")
        return {"error": f"Failed to load model {model_to_use}: {str(e)}"}

    # Convert messages to LangChain format
    langchain_messages = convert_messages_to_langchain(request.messages)

    # Prepare response data for saving
    response_data = ResponseData(
        user_id=headers.user_id or "anonymous",
        team_id=headers.team_id or "default",
        session_id=headers.session_id or f"session-{int(time.time())}",
        project_name=headers.project_name or "default",
        model_used=model_to_use,
        messages=request.messages,
        response="",  # Will be filled later
        timestamp=datetime.now(),
        metadata={
            "temperature": request.temperature,
            "max_tokens": max_tokens,
            "use_case": headers.use_case,
            "priority": headers.priority
        }
    )

    try:
        if request.stream:
            logger.info("   📡 Streaming response via LangChain")
            return StreamingResponse(
                stream_langchain_response(model, langchain_messages, model_to_use, response_data),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Model-Used": model_to_use
                }
            )
        else:
            logger.info("   📄 Non-streaming response via LangChain")
            response = await get_langchain_response(model, langchain_messages, model_to_use, response_data)
            return response

    except Exception as e:
        logger.error(f"   ❌ Error processing request: {e}")
        return {"error": f"Error processing request: {str(e)}"}


@app.get("/")
async def root():
    return {
        "message": "LangChain AI Proxy Server",
        "version": "1.0.0",
        "supported_models": list(MODEL_CONFIGS.keys()),
        "features": [
            "Multi-model routing",
            "Streaming responses",
            "Response saving",
            "Custom headers support"
        ]
    }


@app.get("/models")
async def list_models():
    """List available models and their configurations"""
    return {
        "models": MODEL_CONFIGS,
        "routing_info": "Set X-Model-Preference header or specify model in request"
    }


if __name__ == "__main__":
    import uvicorn

    # Make sure Ollama is running for Gemma
    logger.info("🚀 Starting LangChain AI Proxy Server")
    logger.info("   Make sure Ollama is running: ollama serve")
    logger.info("   And Gemma is pulled: ollama pull gemma:4b")

    uvicorn.run(app, host="0.0.0.0", port=8000)