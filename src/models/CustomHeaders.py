from typing import Optional
from pydantic import BaseModel


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

