from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from .context_manager import ContextManager
from src.helpers.code_analysis_utils import analyze_message_content  # Import code analysis
import logging
from langchain_core.prompts import SystemMessagePromptTemplate

logger = logging.getLogger(__name__)


class ConversationState(TypedDict):
    # Input
    messages: List
    project_name: str
    user_id: str
    session_id: str
    model: str
    headers: Dict  # Added headers for code analysis

    # Context
    similar_contexts: List[dict]
    recent_messages: List[dict]

    # Code analysis results
    message_analysis: Dict
    detected_task_type: str
    detected_language: str

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
        workflow.add_node("analyze_content", self.analyze_content)  # New: analyze code content
        workflow.add_node("search_context", self.search_context)
        workflow.add_node("prepare_prompt", self.prepare_prompt)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("save_interaction", self.save_interaction)

        # Define edges
        workflow.set_entry_point("analyze_content")  # Start with content analysis
        workflow.add_edge("analyze_content", "search_context")
        workflow.add_edge("search_context", "prepare_prompt")
        workflow.add_edge("prepare_prompt", "generate_response")
        workflow.add_edge("generate_response", "save_interaction")
        workflow.add_edge("save_interaction", END)

        return workflow.compile()

    async def analyze_content(self, state: ConversationState) -> ConversationState:
        """Analyze message content for coding-specific metadata"""

        # Extract the current user message
        current_message = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                current_message = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        # Analyze the message content
        headers_obj = state.get("headers", {})
        headers_dict = {
            "language": getattr(headers_obj, "language", None) if headers_obj else None,
            "task_type": getattr(headers_obj, "task_type", None) if headers_obj else None,
            "file_path": getattr(headers_obj, "file_path", None) if headers_obj else None
        }

        analysis = analyze_message_content(current_message, headers_dict)

        # Store analysis results in state
        state["message_analysis"] = analysis
        state["detected_task_type"] = analysis["task_type"]
        state["detected_language"] = analysis["language"]

        logger.info(f"Content analysis: Language={analysis['language']}, Task={analysis['task_type']}, "
                    f"Code blocks={len(analysis['code_blocks'])}, Error-related={analysis['is_error_related']}")

        return state

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

        # Get recent messages from the current session
        recent_messages = await self.context_manager.get_recent_messages(
            project_name=state["project_name"],
            user_id=state["user_id"],
            session_id=state["session_id"],  # Filter by session_id
            limit=3
        )

        state["similar_contexts"] = similar_contexts
        state["recent_messages"] = recent_messages
        state["context_used"] = bool(similar_contexts or recent_messages)

        logger.info(
            f"Found {len(similar_contexts)} similar contexts and {len(recent_messages)} recent messages in session {state['session_id']}")

        return state

    async def prepare_prompt(self, state: ConversationState) -> ConversationState:
        """Inject context into the conversation using LangChain templates"""

        # Get analysis results
        analysis = state.get("message_analysis", {})
        task_type = state.get("detected_task_type", "general")
        language = state.get("detected_language", "python")

        # Define the enhanced context template for coding
        context_template = SystemMessagePromptTemplate.from_template(
            """You are an expert {language} developer and coding assistant.
            You have access to previous conversations from this project that might be relevant.

            Project: {project_name}
            Session: {session_id}
            Task Type: {task_type}
            Language: {language}
            ==================================================

            {similar_contexts}
            {recent_history}
            {code_context}

            Instructions:
            - Focus on {language} best practices and conventions
            - For {task_type} tasks, prioritize accuracy and clarity
            - Reference previous solutions when relevant
            - Include code examples when helpful
            - Consider error patterns and debugging context when applicable

            Please provide helpful, accurate responses while maintaining consistency with previous answers."""
        )

        # Build similar contexts section
        similar_contexts_text = self._format_similar_contexts(state["similar_contexts"])

        # Build recent history section
        recent_history_text = self._format_recent_history(state["recent_messages"])

        # Build code-specific context
        code_context_text = self._format_code_context(analysis)

        # Only inject context if we have meaningful content
        if similar_contexts_text or recent_history_text or code_context_text:
            # Format the system message using the template
            system_message = context_template.format(
                project_name=state["project_name"],
                session_id=state["session_id"],
                task_type=task_type,
                language=language,
                similar_contexts=similar_contexts_text,
                recent_history=recent_history_text,
                code_context=code_context_text
            )

            # Insert or update system message
            if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
                state["messages"].insert(0, system_message)
            else:
                # Append to existing system message
                state["messages"][0].content += f"\n\n{system_message.content}"

        return state

    def _format_code_context(self, analysis: Dict) -> str:
        """Format code analysis results into readable context"""
        if not analysis:
            return ""

        lines = ["\n### Code Context for Current Request:"]

        # Add file information
        file_info = analysis.get("file_info", {})
        if file_info.get("file_paths"):
            lines.append(f"Files mentioned: {', '.join(file_info['file_paths'][:3])}")  # Limit to 3 files

        # Add libraries/imports
        imports = analysis.get("imports", {})
        if imports.get("all_libraries"):
            libs = imports["all_libraries"][:5]  # Limit to 5 libraries
            lines.append(f"Libraries involved: {', '.join(libs)}")

        # Add functions mentioned
        functions = analysis.get("functions", [])
        if functions:
            funcs = functions[:3]  # Limit to 3 functions
            lines.append(f"Functions/Classes: {', '.join(funcs)}")

        # Add complexity and error context
        if analysis.get("is_error_related"):
            lines.append("⚠️ This appears to be related to debugging/error resolution")

        complexity = analysis.get("complexity_score", 0)
        if complexity > 20:
            lines.append(f"💡 High complexity conversation (score: {complexity})")

        return "\n".join(lines) if len(lines) > 1 else ""

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

        lines = ["\n### Recent Conversation History in this Session:"]

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
        """Save the interaction to the database with enhanced coding metadata"""

        # Extract the user's question
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if user_message and state.get("final_response"):
            # Get analysis results
            analysis = state.get("message_analysis", {})
            headers_obj = state.get("headers", {})

            # Enhanced metadata with code analysis
            user_metadata = {
                "context_used": state.get("context_used", False),
                "language": state.get("detected_language", "python"),
                "task_type": state.get("detected_task_type", "general"),
                "file_path": getattr(headers_obj, "file_path", None) if headers_obj else None,
                "code_analysis": analysis,
                "has_code_blocks": analysis.get("has_code", False),
                "is_error_related": analysis.get("is_error_related", False),
                "complexity_score": analysis.get("complexity_score", 0)
            }

            # Save user message with enhanced metadata
            user_msg_id = await self.context_manager.save_message(
                project_name=state["project_name"],
                user_id=state["user_id"],
                session_id=state["session_id"],
                role="user",
                content=user_message,
                metadata=user_metadata
            )

            # Enhanced metadata for assistant response
            assistant_metadata = {
                "context_used": state.get("context_used", False),
                "similar_contexts_count": len(state.get("similar_contexts", [])),
                "recent_messages_count": len(state.get("recent_messages", [])),
                "language": state.get("detected_language", "python"),
                "task_type": state.get("detected_task_type", "general"),
                "user_had_code": analysis.get("has_code", False),
                "user_had_errors": analysis.get("is_error_related", False),
                "complexity_score": analysis.get("complexity_score", 0)
            }

            # Save assistant response with enhanced metadata
            await self.context_manager.save_message(
                project_name=state["project_name"],
                user_id=state["user_id"],
                session_id=state["session_id"],
                role="assistant",
                content=state["final_response"],
                model_used=state["model"],
                parent_message_id=user_msg_id,
                metadata=assistant_metadata
            )

            logger.info(f"Saved enhanced Q&A pair for project {state['project_name']}, "
                        f"session {state['session_id']}, language {state.get('detected_language', 'python')}")

        return state

    async def process(self,
                      messages: List,
                      project_name: str,
                      user_id: str,
                      session_id: str,
                      model: str,
                      headers: Dict = None) -> ConversationState:  # Added headers parameter
        """Process a conversation through the workflow"""

        initial_state = ConversationState(
            messages=messages,
            project_name=project_name,
            user_id=user_id,
            session_id=session_id,
            model=model,
            headers=headers or {},  # Include headers in state
            similar_contexts=[],
            recent_messages=[],
            message_analysis={},  # Initialize analysis fields
            detected_task_type="general",
            detected_language="python",
            final_response="",
            context_used=False
        )

        result = await self.workflow.ainvoke(initial_state)
        return result