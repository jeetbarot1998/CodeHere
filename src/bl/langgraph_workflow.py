from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from .context_manager import ContextManager
import logging
from langchain_core.prompts import SystemMessagePromptTemplate

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    # Input
    messages: List
    project_name: str
    user_id: str
    model: str

    # Context
    similar_contexts: List[dict]
    recent_messages: List[dict]

    # Output
    final_response: str
    context_used: bool


class ContextAwareWorkflow:
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ConversationState)

        # Add nodes
        workflow.add_node("search_context", self.search_context)
        workflow.add_node("prepare_prompt", self.prepare_prompt)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("save_interaction", self.save_interaction)

        # Define edges
        workflow.set_entry_point("search_context")
        workflow.add_edge("search_context", "prepare_prompt")
        workflow.add_edge("prepare_prompt", "generate_response")
        workflow.add_edge("generate_response", "save_interaction")
        workflow.add_edge("save_interaction", END)

        return workflow.compile()

    async def search_context(self, state: ConversationState) -> ConversationState:
        """Search for similar context and recent messages"""

        # Extract the current query from messages
        current_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                current_query = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        # Search similar contexts
        similar_contexts = await self.context_manager.search_similar_context(
            query=current_query,
            project_name=state["project_name"],
            limit=3
        )

        # Get recent messages
        recent_messages = await self.context_manager.get_recent_messages(
            project_name=state["project_name"],
            user_id=state["user_id"],
            limit=3
        )

        state["similar_contexts"] = similar_contexts
        state["recent_messages"] = recent_messages
        state["context_used"] = bool(similar_contexts or recent_messages)

        logger.info(f"Found {len(similar_contexts)} similar contexts and {len(recent_messages)} recent messages")

        return state

    async def prepare_prompt(self, state: ConversationState) -> ConversationState:
        """Inject context into the conversation using LangChain templates"""

        # Define the context template
        context_template = SystemMessagePromptTemplate.from_template(
            """You are part of an AI development team assistant. 
            You have access to previous conversations from this project that might be relevant.
        
            Project: {project_name}
            ==================================================
        
            {similar_contexts}
            {recent_history}
        
            Please consider this context when answering, but only reference it if directly relevant.
            Maintain consistency with previous answers when applicable."""
        )

        # Build similar contexts section
        similar_contexts_text = self._format_similar_contexts(state["similar_contexts"])

        # Build recent history section
        recent_history_text = self._format_recent_history(state["recent_messages"])

        # Only inject context if we have meaningful content
        if similar_contexts_text or recent_history_text:
            # Format the system message using the template
            system_message = context_template.format(
                project_name=state["project_name"],
                similar_contexts=similar_contexts_text,
                recent_history=recent_history_text
            )

            # Insert or update system message
            if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
                state["messages"].insert(0, system_message)
            else:
                # Append to existing system message
                state["messages"][0].content += f"\n\n{system_message.content}"

        return state

    def _format_similar_contexts(self, similar_contexts: List[Dict]) -> str:
        """Format similar contexts into readable text"""
        if not similar_contexts:
            return ""

        lines = ["### Similar Previous Conversations in this Project:"]

        for i, ctx in enumerate(similar_contexts, 1):
            lines.extend([
                f"\n**Example {i}** (relevance: {ctx['score']:.2f}):",
                f"Q: {ctx['question']}",
                f"A: {ctx['answer']}"
            ])

        return "\n".join(lines)

    def _format_recent_history(self, recent_messages: List[Dict]) -> str:
        """Format recent conversation history into readable text"""
        if not recent_messages:
            return ""

        lines = ["\n### Recent Conversation History:"]

        for msg in recent_messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")

        return "\n".join(lines)

    async def generate_response(self, state: ConversationState) -> ConversationState:
        """Generate response (placeholder - will be handled by main app)"""
        # This is just a placeholder - actual generation happens in the main FastAPI app
        # We're just passing through the messages with context injected
        return state

    async def save_interaction(self, state: ConversationState) -> ConversationState:
        """Save the interaction to the database"""

        # Extract the user's question
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if user_message and state.get("final_response"):
            # Save user message
            user_msg_id = await self.context_manager.save_message(
                project_name=state["project_name"],
                user_id=state["user_id"],
                role="user",
                content=user_message,
                metadata={"context_used": state.get("context_used", False)}
            )

            # Save assistant response with link to user message
            await self.context_manager.save_message(
                project_name=state["project_name"],
                user_id=state["user_id"],
                role="assistant",
                content=state["final_response"],
                model_used=state["model"],
                parent_message_id=user_msg_id,
                metadata={
                    "context_used": state.get("context_used", False),
                    "similar_contexts_count": len(state.get("similar_contexts", [])),
                    "recent_messages_count": len(state.get("recent_messages", []))
                }
            )

            logger.info(f"Saved Q&A pair for project {state['project_name']}")

        return state

    async def process(self,
                      messages: List,
                      project_name: str,
                      user_id: str,
                      model: str) -> ConversationState:
        """Process a conversation through the workflow"""

        initial_state = ConversationState(
            messages=messages,
            project_name=project_name,
            user_id=user_id,
            model=model,
            similar_contexts=[],
            recent_messages=[],
            final_response="",
            context_used=False
        )

        result = await self.workflow.ainvoke(initial_state)
        return result