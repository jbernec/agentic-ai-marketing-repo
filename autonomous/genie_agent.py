"""
Autonomous Genie Agent with validation and HITL review.

This agent adds validation and human-in-the-loop review logic
around Databricks Genie queries.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langchain_core.messages import HumanMessage, SystemMessage
from .base_agent import AutonomousAgent
from .hitl_utils import requires_review


class AutonomousGenieAgent(AutonomousAgent):
    """
    Autonomous agent for Databricks Genie with validation and review.
    
    Internal workflow:
    1. Entry: Extract and validate user query
    2. Validate: Check if query is appropriate for Genie
    3. Execute: Run Genie query with summarized messages
    4. Review: Check if results need human review (HITL)
    5. Exit: Return results (possibly after human approval)
    
    Benefits over direct Genie access:
    - Validates queries before expensive API calls
    - Handles message truncation automatically
    - Implements HITL for sensitive operations
    - Better error handling and recovery
    """
    
    def __init__(self, model, genie_agent):
        """
        Initialize Genie agent.
        
        Args:
            model: LLM model
            genie_agent: GenieAgent instance from databricks_langchain
        """
        super().__init__(
            name="Genie",
            model=model,
            tools=[],  # Genie is not a tool, it's invoked directly
            verbose=True
        )
        self.genie = genie_agent
    
    # ==================== REQUIRED IMPLEMENTATIONS ====================
    
    def _build_graph(self, graph: StateGraph) -> StateGraph:
        """Define Genie-specific workflow - simplified"""
        # Single execution node
        graph.add_node("execute", self._execute_genie)
        
        # Simple linear flow: entry -> execute -> exit
        graph.add_edge("entry", "execute")
        graph.add_edge("execute", "exit")
        
        return graph
    
    def _entry_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user query - simplified"""
        return {}  # Just pass through
    
    def _execute_genie(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Genie query with pre-execution HITL check"""
        self._log("Executing Genie query")
        
        # Extract user query
        messages = state.get("messages", [])
        user_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
        
        # Check if user is requesting a destructive operation BEFORE calling Genie
        if user_query and requires_review(user_query):
            self._log("Destructive operation detected - requesting approval")
            # Interrupt to get approval before execution
            approved = interrupt({
                "query": user_query,
                "message": "⚠️ **Destructive Operation Detected**\n\nYou're requesting an operation that will modify or delete data. Do you want to proceed?",
                "operation_type": "pre_execution_approval"
            })
            # If user didn't approve, return early
            if not approved:
                self._log("Operation cancelled by user")
                return {
                    "messages": [{
                        "role": "assistant",
                        "name": self.name,
                        "content": "Operation cancelled by user."
                    }]
                }
        
        # User approved or it's a safe query - proceed with Genie
        messages_to_send = state.get("summarized_messages", state.get("messages", []))
        
        try:
            response = self.genie.invoke({"messages": messages_to_send})
            content = response["messages"][-1].content
            
            self._log(f"Genie returned: {len(content)} chars")
            
            # Return directly in messages format
            return {
                "messages": [{
                    "role": "assistant",
                    "name": self.name,
                    "content": content
                }]
            }
        except Exception as e:
            self._log(f"Genie failed: {e}", level="error")
            return {
                "messages": [{
                    "role": "assistant",
                    "name": self.name,
                    "content": f"Error: {str(e)}"
                }]
            }
    
    def _exit_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Exit node - messages already set"""
        return {}
