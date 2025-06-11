from typing import List, Union, Optional
from pydantic import BaseModel

from src.models.Message import Message


class ChatRequest(BaseModel):
    messages: List[Message]
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = True
