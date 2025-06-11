from pydantic import BaseModel

class TextContent(BaseModel):
    type: str
    text: str
