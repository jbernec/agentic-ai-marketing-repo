"""
Base class for autonomous agents with internal sub-graphs.

Provides common infrastructure for agents that make their own internal decisions
about tool usage, routing, and execution flow.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
import time
import logging


class AutonomousAgent(ABC):
    """
    Base class for autonomous agents with internal sub-graphs.
    
    Provides common infrastructure while allowing agent-specific logic.
    
    Key Features:
    - Automatic timing and logging for all nodes
    - Common tool execution pattern
    - Entry/exit node wrappers
    - Helper methods for state management
    
    Subclasses must implement:
    - _build_graph(): Define agent-specific nodes and routing
    - _entry_node(): First node logic (validation, setup)
    - _exit_node(): Final node logic (format results)
    """
    
    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        tools: Optional[List[BaseTool]] = None,
        verbose: bool = True
    ):
        """
        Initialize autonomous agent.
        
        Args:
            name: Agent name (for logging and identification)
            model: LLM model to use
            tools: Optional list of tools to bind to model
            verbose: Enable debug logging
        """
        self.name = name
        self.model = model
        self.tools = tools or []
        self.model_with_tools = model.bind_tools(self.tools) if self.tools else model
        self.verbose = verbose
        self.logger = logging.getLogger(f"Agent.{name}")
        self._compiled_graph = None
    
    # ==================== ABSTRACT METHODS (Must Implement) ====================
    
    @abstractmethod
    def _build_graph(self, graph: StateGraph) -> StateGraph:
        """
        Add nodes and edges to the graph.
        
        Subclasses implement their specific workflow here.
        Entry and exit nodes are automatically added.
        
        Example:
            graph.add_node("plan", self._plan)
            graph.add_node("execute", self._execute)
            graph.add_edge("entry", "plan")
            graph.add_edge("plan", "execute")
            graph.add_edge("execute", "exit")
            return graph
        
        Args:
            graph: StateGraph to populate
            
        Returns:
            Modified StateGraph
        """
        pass
    
    @abstractmethod
    def _entry_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        First node executed - agent-specific setup/validation.
        
        Use this for:
        - Validating input
        - Extracting user query
        - Initial classification
        - Setting up agent-specific state
        
        Args:
            state: Current state
            
        Returns:
            Updated state dict
        """
        pass
    
    @abstractmethod
    def _exit_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Final node - format and return results.
        
        Must return state with 'messages' key containing final response.
        
        Args:
            state: Current state
            
        Returns:
            Updated state dict with messages key
        """
        pass
    
    # ==================== COMMON METHODS (Provided) ====================
    
    def create_agent(self) -> CompiledStateGraph:
        """
        Build and compile the agent's graph.
        
        This is the public interface - supervisor calls this.
        Automatically adds entry/exit nodes with timing wrappers.
        
        Returns:
            Compiled StateGraph ready for execution
        """
        if self._compiled_graph is None:
            # Import here to avoid circular dependencies
            import sys
            import os
            # Import from autonomous.graph, not the parent graph module
            from .graph import AgentState
            
            graph = StateGraph(AgentState)
            
            # All agents get entry and exit nodes
            graph.add_node("entry", self._entry_node_wrapper)
            graph.add_node("exit", self._exit_node_wrapper)
            
            # Let subclass add its specific nodes
            graph = self._build_graph(graph)
            
            # Connect entry and exit
            graph.add_edge(START, "entry")
            # Subclass handles middle routing and must eventually route to "exit"
            graph.add_edge("exit", END)
            
            # Import memory_setup to get store for compilation
            # Sub-graphs need store to be compiled with it for nested nodes to access it
            import memory_setup
            store = memory_setup.get_store()
            
            self._compiled_graph = graph.compile(store=store)
            
        return self._compiled_graph
    
    def _entry_node_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper that adds timing and logging to entry"""
        start_time = time.time()
        self._log(f"Entering {self.name} agent")
        
        result = self._entry_node(state)
        
        execution_times = result.get("execution_times", {})
        execution_times[f"{self.name}_entry"] = time.time() - start_time
        result["execution_times"] = execution_times
        
        return result
    
    def _exit_node_wrapper(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper that adds timing and logging to exit"""
        start_time = time.time()
        
        result = self._exit_node(state)
        
        execution_times = result.get("execution_times", state.get("execution_times", {}))
        execution_times[f"{self.name}_exit"] = time.time() - start_time
        result["execution_times"] = execution_times
        
        self._log(f"Exiting {self.name} agent")
        return result
    
    def execute_tools(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Common tool execution logic - all agents use this.
        
        Handles tool calls from LLM responses, executes them,
        and returns results in state.
        
        Args:
            state: State with tool_calls key
            
        Returns:
            State with tool_results key
        """
        if not state.get("tool_calls"):
            return {"tool_results": []}
        
        results = []
        for tool_call in state["tool_calls"]:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            tool = next((t for t in self.tools if t.name == tool_name), None)
            
            if tool:
                try:
                    self._log(f"Executing tool: {tool_name} with args: {tool_args}")
                    result = tool.invoke(tool_args)
                    results.append(result)
                except Exception as e:
                    self._log(f"Tool execution failed: {e}", level="error")
                    results.append({"error": str(e)})
            else:
                self._log(f"Tool not found: {tool_name}", level="warning")
        
        return {"tool_results": results}
    
    def extract_user_query(self, state: Dict[str, Any]) -> str:
        """
        Common helper - extract last user message from state.
        
        Args:
            state: State with messages
            
        Returns:
            User query string or empty string
        """
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content
        return ""
    
    def track_execution(self, node_name: str):
        """
        Decorator for tracking node execution time.
        
        Usage:
            @agent.track_execution("planning")
            def _plan(self, state):
                ...
        
        Args:
            node_name: Name to use in execution_times dict
        """
        def decorator(func):
            def wrapper(state):
                start_time = time.time()
                result = func(state)
                
                execution_times = result.get("execution_times", state.get("execution_times", {}))
                execution_times[f"{self.name}_{node_name}"] = time.time() - start_time
                result["execution_times"] = execution_times
                
                return result
            return wrapper
        return decorator
    
    def _log(self, message: str, level: str = "info"):
        """
        Common logging helper.
        
        Args:
            message: Message to log
            level: Log level (info, warning, error)
        """
        if self.verbose:
            getattr(self.logger, level)(message)
    
    # ==================== OPTIONAL HOOKS (Override if Needed) ====================
    
    def validate_input(self, state: Dict[str, Any]) -> bool:
        """
        Override to add input validation.
        
        Returns:
            True if input is valid
        """
        return True
    
    def requires_review(self, state: Dict[str, Any]) -> bool:
        """
        Override to add HITL (Human-in-the-Loop) logic.
        
        Returns:
            True if human review is needed
        """
        return False
