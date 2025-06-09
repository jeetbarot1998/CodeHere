from src.models.CustomHeaders import CustomHeaders
from fastapi import Header
from typing import Optional

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
