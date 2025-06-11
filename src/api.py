from fastapi import FastAPI, Request, Header, Depends
from pydantic import BaseModel
from typing import List, Union, Optional, AsyncGenerator
import time
import logging
from fastapi.responses import StreamingResponse
import json
import asyncio
import os
from contextlib import asynccontextmanager
from helpers.artifact_handler import ArtifactHandler  # Import artifact handler

# LangChain imports
from langchain_ollama import OllamaLLM as Ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Context management imports
from bl.context_manager import ContextManager
from bl.langgraph_workflow import ContextAwareWorkflow, ConversationState
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global context manager and workflow
context_manager = None
context_workflow : ContextAwareWorkflow = None
artifact_handler = ArtifactHandler(size_threshold=500)  # Initialize artifact handler

load_dotenv('.env', override=True)  # Force load from specific file


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup context manager"""
    global context_manager, context_workflow

    # Startup
    context_manager = ContextManager()
    await context_manager.init_postgres()
    context_workflow = ContextAwareWorkflow(context_manager)
    logger.info("✅ Context manager initialized")

    yield

    # Shutdown
    await context_manager.close()
    logger.info("👋 Context manager closed")


app = FastAPI(
    title="LangChain AI Proxy Server with Context",
    version="2.0.0",
    lifespan=lifespan
)


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
    # Code-specific headers
    language: Optional[str] = None  # Programming language
    task_type: Optional[str] = None  # debug, implement, explain, review, refactor
    file_path: Optional[str] = None  # File being discussed


# Model configuration (same as before)
MODEL_CONFIGS = {
    "gemma": {
        "type": "ollama",
        "model_name": "gemma3:4B",
        "base_url": "http://localhost:11434"
    },
    "gpt-4": {
        "type": "openai",
        "model_name": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY")  # Set via environment variable
    },
    "claude": {
        "type": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "api_key": os.getenv("ANTHROPIC_API_KEY")  # Set via environment variable
    }
}


def get_langchain_model(model_name: str, temperature: float = 0.7, max_tokens: int = 2048):
    """Factory function to create LangChain model instances"""
    config = MODEL_CONFIGS.get(model_name.lower())

    if not config:
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
            api_key=config.get("api_key")
        )
    elif config["type"] == "anthropic":
        return ChatAnthropic(
            model=config["model_name"],
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=config.get("api_key")
        )
    else:
        raise ValueError(f"Unsupported model type: {config['type']}")


def convert_messages_to_langchain(model_to_use, messages: List[Message]) -> List:
    """Convert OpenAI format messages to LangChain format with artifact processing"""
    global artifact_handler
    langchain_messages = []

    for message in messages:
        content = extract_text_content(message.content)

        # Process content for artifacts if it's a user message
        if message.role == "user":
            # Do this artifact only for claude.
            if model_to_use == "anthropic":
                # Check for file contents that should be artifacts
                cleaned_content, artifacts = artifact_handler.process_message_for_artifacts(content)

                # Add artifacts as separate messages before the user message
                for artifact in artifacts:
                    langchain_messages.append(HumanMessage(content=artifact["text"]))

                # Add the cleaned user message
                langchain_messages.append(HumanMessage(content=cleaned_content))
            else:
                langchain_messages.append(HumanMessage(content=content))

        elif message.role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        elif message.role == "system":
            langchain_messages.append(SystemMessage(content=content))

    return langchain_messages


def extract_text_content(content: Union[str, List[TextContent]]) -> str:
    """Extract text from content"""
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


def route_model_by_preference(headers: CustomHeaders, request_model: str) -> str:
    """Route to appropriate model based on headers and request"""
    if request_model and request_model.lower() in MODEL_CONFIGS:
        return request_model.lower()

    if headers.use_case == "quick_answers" or headers.priority == "fast":
        return "gemma"
    elif headers.model_preference == "anthropic":
        return "claude"
    elif headers.model_preference == "openai":
        return "gpt-4"
    else:
        return "gemma"


async def stream_langchain_response_with_context(
        model,
        messages: List,
        model_name: str,
        workflow_state: dict
) -> AsyncGenerator[str, None]:
    """Stream response from LangChain model with context"""
    try:
        full_response = ""

        # For Ollama models
        if isinstance(model, Ollama):
            response = await asyncio.to_thread(
                lambda: model.invoke(messages)
            )

            words = response.split()

            for i, word in enumerate(words):
                chunk_content = word + " " if i < len(words) - 1 else word
                full_response += chunk_content

                chunk = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": chunk_content},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        else:
            # For OpenAI/Anthropic models with proper streaming
            async for chunk in model.astream(messages):
                full_response += chunk.content
                chunk_data = {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": chunk.content},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

        # Save the complete response to context
        if context_workflow and workflow_state:
            workflow_state["final_response"] = full_response.strip()
            await context_workflow.save_interaction(workflow_state)

        # Final chunk
        final_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        error_chunk = {"error": {"message": str(e), "type": "api_error"}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


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
        x_use_case: Optional[str] = Header(None, alias="X-Use-Case"),
        # Code-specific headers
        x_language: Optional[str] = Header(None, alias="X-Language"),
        x_task_type: Optional[str] = Header(None, alias="X-Task-Type"),
        x_file_path: Optional[str] = Header(None, alias="X-File-Path")
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
        use_case=x_use_case,
        language=x_language,
        task_type=x_task_type,
        file_path=x_file_path
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"🔥 REQUEST: {request.method} {request.url.path}")
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
    logger.info(f"🔧 Context-Aware Request from User: {headers.user_id}")
    logger.info(f"   Project: {headers.project_name}")
    logger.info(f"   Session: {headers.session_id}")
    logger.info(f"   Language: {headers.language}")  # Log programming language
    logger.info(f"   Task Type: {headers.task_type}")  # Log task type

    # Set defaults
    project_name = headers.project_name or "default"
    user_id = headers.user_id or "anonymous"
    session_id = headers.session_id or "default_session"

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
    langchain_messages = convert_messages_to_langchain(model_to_use, request.messages)
    workflow_state = None

    # Process through context workflow
    if context_workflow:
        logger.info("   🧠 Processing with context awareness")

        # Run the workflow to get context-enhanced messages
        workflow_state: ConversationState = await context_workflow.process(
            messages=langchain_messages,
            project_name=project_name,
            user_id=user_id,
            session_id=session_id,
            model=model_to_use,
            headers=headers  # Pass the CustomHeaders object directly
        )

        # Use the context-enhanced messages
        langchain_messages: List = workflow_state["messages"]

        logger.info(f"   📚 Context injected: {workflow_state['context_used']}")

    try:
        if request.stream:
            logger.info("   📡 Streaming response with context")

            return StreamingResponse(
                stream_langchain_response_with_context(
                    model, langchain_messages, model_to_use, workflow_state
                ),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Model-Used": model_to_use,
                    "X-Context-Used": str(workflow_state.get("context_used", False)),
                    "X-Session-ID": session_id,
                    "X-Language": headers.language or "python",  # Return detected language
                    "X-Task-Type": workflow_state.get("detected_task_type", "general") if workflow_state else "general",
                    "X-Artifacts-Used": "true"  # Indicate artifacts were processed
                }
            )
        else:
            logger.info("   📄 Non-streaming response with context")

            # Get response
            response = await asyncio.to_thread(
                lambda: model.invoke(langchain_messages)
            )
            response_text = response if isinstance(response, str) else str(response)

            # Save to context
            if context_workflow:
                workflow_state["final_response"] = response_text
                await context_workflow.save_interaction(workflow_state)

            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_to_use,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": 10 + len(response_text.split())
                }
            }

    except Exception as e:
        logger.error(f"   ❌ Error processing request: {e}")
        return {"error": f"Error processing request: {str(e)}"}


@app.post("/star/{message_id}")
async def star_message(message_id: str):
    """Star a message for quality curation"""
    if context_manager:
        await context_manager.star_message(message_id)
        return {"status": "success", "message_id": message_id}
    return {"error": "Context manager not initialized"}


@app.get("/projects/{project_name}/starred")
async def get_starred_messages(project_name: str):
    """Get starred messages for a project"""
    if context_manager:
        messages = await context_manager.get_starred_messages(project_name)
        return {"project": project_name, "starred_messages": messages}
    return {"error": "Context manager not initialized"}


@app.get("/projects/{project_name}/code-stats")
async def get_project_code_stats(project_name: str):
    """Get coding statistics for a project"""
    if context_manager:
        stats = await context_manager.get_project_code_statistics(project_name)
        return {"project": project_name, "code_statistics": stats}
    return {"error": "Context manager not initialized"}


@app.get("/projects/{project_name}/libraries")
async def get_project_libraries(project_name: str):
    """Get most used libraries in a project"""
    if context_manager:
        libraries = await context_manager.get_project_libraries(project_name)
        return {"project": project_name, "libraries": libraries}
    return {"error": "Context manager not initialized"}


@app.post("/tag/{message_id}")
async def tag_message(message_id: str, tag: str = "helpful"):
    """Tag a message for training data curation"""
    if context_manager:
        await context_manager.tag_message(message_id, tag)
        return {"status": "success", "message_id": message_id, "tag": tag}
    return {"error": "Context manager not initialized"}


@app.get("/sessions/{project_name}/{user_id}/{session_id}")
async def get_session_messages(project_name: str, user_id: str, session_id: str):
    """Get all messages from a specific session"""
    if context_manager:
        messages = await context_manager.get_session_messages(project_name, user_id, session_id)
        return {
            "project": project_name,
            "user_id": user_id,
            "session_id": session_id,
            "messages": messages
        }
    return {"error": "Context manager not initialized"}


@app.post("/test-artifacts")
async def test_artifacts(content: str):
    """Test endpoint to see how content would be processed for artifacts"""
    global artifact_handler

    cleaned_content, artifacts = artifact_handler.process_message_for_artifacts(content)

    return {
        "original_length": len(content),
        "cleaned_content": cleaned_content,
        "cleaned_length": len(cleaned_content),
        "artifacts_count": len(artifacts),
        "artifacts": [
            {
                "size": len(artifact["text"]),
                "preview": artifact["text"][:200] + "..." if len(artifact["text"]) > 200 else artifact["text"]
            }
            for artifact in artifacts
        ]
    }


@app.get("/")
async def root():
    return {
        "message": "LangChain AI Proxy Server with Context",
        "version": "2.0.0",
        "supported_models": list(MODEL_CONFIGS.keys()),
        "features": [
            "Multi-model routing",
            "Streaming responses",
            "Context-aware responses",
            "Project-based knowledge",
            "Session-based conversations",
            "Vector similarity search",
            "Message starring",
            "LangGraph workflows",
            "Automatic artifact processing",  # Added artifact feature
            "Code-aware metadata collection"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "context_manager": context_manager is not None,
        "context_workflow": context_workflow is not None
    }

    # Check database connections
    if context_manager:
        try:
            # Test PostgreSQL
            async with context_manager.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            health_status["postgresql"] = "connected"
        except:
            health_status["postgresql"] = "disconnected"
            health_status["status"] = "degraded"

        try:
            # Test Qdrant
            context_manager.qdrant_client.get_collections()
            health_status["qdrant"] = "connected"
        except:
            health_status["qdrant"] = "disconnected"
            health_status["status"] = "degraded"

    return health_status


if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting Context-Aware LangChain AI Proxy Server")
    logger.info("   Make sure Docker services are running:")
    logger.info("   docker-compose up -d")
    logger.info("   And Ollama models are available:")
    logger.info("   ollama pull gemma3:4b")
    logger.info("   ollama pull nomic-embed-text")

    uvicorn.run(app, host="0.0.0.0", port=8000)