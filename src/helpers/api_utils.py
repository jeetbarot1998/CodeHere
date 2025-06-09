from src.models.CustomHeaders import CustomHeaders
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from typing import List, Union
# LangChain imports
from langchain_ollama import OllamaLLM as Ollama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from src.models.Message import Message
from src.models.TextContent import TextContent

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

