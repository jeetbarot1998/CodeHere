from typing import List, Union
from pydantic import BaseModel
from src.models.TextContent import TextContent


class Message(BaseModel):
    role: str  # "user" or "assistant" or "system"
    content: Union[str, List[TextContent]]

