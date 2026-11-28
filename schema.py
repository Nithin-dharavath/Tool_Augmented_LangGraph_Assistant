from pydantic import BaseModel
from typing import TypedDict, Annotated, Optional

class WorkflowState(TypedDict):
    messages: Annotated[list, add_messages]
    summary : str


class ChatRequest(BaseModel):
    human_message: str
    thread_id: Optional[str] = None