"""
Autonomous Memory Agent with internal classification logic.

This agent decides internally whether user is sharing information (store)
or asking about past information (recall).
"""

from typing import Dict, Any
from langgraph.graph import StateGraph
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from .base_agent import AutonomousAgent


class AutonomousMemoryAgent(AutonomousAgent):
    """
    Autonomous agent for memory storage and recall with classification.
    
    Internal workflow:
    1. Entry: Extract user query
    2. Classify: LLM decides if this is "store" or "recall" intent
    3a. Store: Acknowledge new information (actual storage done externally)
    3b. Recall: Search memory store and generate response
    4. Exit: Return formatted response
    
    Benefits over direct memory access:
    - Intelligent classification of user intent
    - Can handle ambiguous queries
    - Adapts response style based on context
    """
    
    def __init__(self, model):
        """
        Initialize memory agent.
        
        Args:
            model: LLM model (no tools needed)
        """
        super().__init__(
            name="Memory",
            model=model,
            tools=[],  # No tools - uses store directly
            verbose=True
        )
    
    # ==================== REQUIRED IMPLEMENTATIONS ====================
    
    def _build_graph(self, graph: StateGraph) -> StateGraph:
        """Define memory-specific workflow"""
        # Add internal nodes
        graph.add_node("classify", self._classify_intent)
        graph.add_node("store", self._store_memory)
        graph.add_node("recall", self._recall_memory)
        
        # Define routing
        graph.add_edge("entry", "classify")
        graph.add_conditional_edges(
            "classify",
            lambda state: state.get("memory_action", "recall"),
            {
                "store": "store",
                "recall": "recall"
            }
        )
        graph.add_edge("store", "exit")
        graph.add_edge("recall", "exit")
        
        return graph
    
    def _entry_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user query"""
        user_query = self.extract_user_query(state)
        self._log(f"Memory query received: {user_query}")
        return {"user_query": user_query}
    
    def _exit_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Format and return memory response"""
        # Debug: Log entire state keys
        self._log(f"Exit node - State keys: {list(state.keys())}")
        self._log(f"Exit node - memory_response in state: {'memory_response' in state}")
        
        content = state.get("memory_response", "Memory operation completed.")
        self._log(f"Exit node - content: {content[:200] if len(content) > 200 else content}")
        
        return {
            "messages": [{
                "role": "assistant",
                "name": self.name,
                "content": content
            }]
        }
    
    # ==================== AGENT-SPECIFIC NODES ====================
    
    def _classify_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent decides if user is sharing info or asking for recall.
        
        Classification logic:
        - "My name is X" → store
        - "What's my name?" → recall
        - "I prefer Y" → store
        - "Do I like Z?" → recall
        """
        import time
        start_time = time.time()
        
        user_query = state.get("user_query", "")
        if not user_query:
            self._log("Warning: No user_query in state for classification")
            return {"memory_action": "recall"}
        
        classification_prompt = f"""Classify this user message into one of two categories:

User message: "{user_query}"

Categories:
A) STORE - User is sharing NEW personal information, preferences, or facts about themselves
   Examples: "My name is John", "I work at Microsoft", "I prefer coffee over tea"
   
B) RECALL - User is asking ABOUT past information or preferences
   Examples: "What's my name?", "Where do I work?", "Do I like coffee?"

Respond with ONLY one word: either "store" or "recall"."""
        
        response = self.model.invoke(classification_prompt)
        action = "store" if "store" in response.content.lower() else "recall"
        
        self._log(f"Classified as: {action} for query: {user_query}")
        
        execution_times = state.get("execution_times", {})
        execution_times[f"{self.name}_classification"] = time.time() - start_time
        
        return {
            "memory_action": action,
            "execution_times": execution_times
        }
    
    def _store_memory(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Acknowledge storing new information.
        
        Note: Actual storage happens in write_long_term_memory node
        outside this agent. This just acknowledges the intent.
        """
        import time
        start_time = time.time()
        
        self._log("Storing new memory")
        
        execution_times = state.get("execution_times", {})
        execution_times[f"{self.name}_store"] = time.time() - start_time
        
        return {
            "memory_response": "Got it! I'll remember that for our future conversations.",
            "execution_times": execution_times
        }
    
    def _recall_memory(self, state: Dict[str, Any], config: RunnableConfig, *, store) -> Dict[str, Any]:
        """
        Recall information from memory store.
        Uses the same pattern as the orchestrated graph's recall_memory_node.
        """
        import time
        start_time = time.time()
        
        user_id = config["configurable"]["user_id"]
        user_query = state.get("user_query", "")
        if not user_query:
            return {
                "memory_response": "I don't have any query to search for in memory.",
                "execution_times": {f"{self.name}_recall": time.time() - start_time}
            }
        
        self._log(f"Recalling from memory for query: {user_query}")
        
        # Search across all memory categories (personal_info first for better recall)
        # This matches the orchestrated graph's approach
        all_memories = []
        for category in ["personal_info", "marketing_insights", "optimization", "inference", "data_insights", "miscellaneous"]:
            namespace = ("user", user_id, category)
            memories = store.search(namespace, query=user_query, limit=5)
            all_memories.extend(memories)
        
        # For personal_info queries, also try searching with specific field filters
        query_lower = user_query.lower()
        if any(word in query_lower for word in ['name', 'called', 'who am i']):
            # Search for memories with name field populated
            personal_memories = store.search(
                ("user", user_id, "personal_info"),
                query="",  # Empty query, rely on filter
                filter={"name": {"$exists": True}},
                limit=5
            )
            all_memories.extend(personal_memories)
        elif any(word in query_lower for word in ['live', 'location', 'where', 'city']):
            # Search for memories with location field
            personal_memories = store.search(
                ("user", user_id, "personal_info"),
                query="",
                filter={"location": {"$exists": True}},
                limit=5
            )
            all_memories.extend(personal_memories)
        
        # Filter out unhelpful "not recorded" memories if we have better ones
        filtered_memories = []
        not_recorded_memories = []
        
        for mem in all_memories:
            summary = mem.value.get('summary', '').lower()
            # Check if this is a "not recorded" or "not in memory" type response
            if any(phrase in summary for phrase in ['not recorded', 'not in', 'is not', "don't have", 'no information']):
                not_recorded_memories.append(mem)
            else:
                filtered_memories.append(mem)
        
        # If we have actual information memories, use only those
        # Otherwise fall back to the "not recorded" ones
        memories_to_use = filtered_memories if filtered_memories else not_recorded_memories
        
        # Format memories for LLM - prioritize structured fields for personal_info
        if memories_to_use:
            memory_details = []
            for i, mem in enumerate(memories_to_use[:5], 1):
                summary = mem.value.get('summary', '')
                text = mem.value.get('text', '')
                created = mem.value.get('created_at', '')
                
                # Add structured fields if available (personal_info)
                structured_info = []
                if mem.value.get('name'):
                    structured_info.append(f"Name: {mem.value['name']}")
                if mem.value.get('location'):
                    structured_info.append(f"Location: {mem.value['location']}")
                if mem.value.get('company'):
                    structured_info.append(f"Company: {mem.value['company']}")
                if mem.value.get('role'):
                    structured_info.append(f"Role: {mem.value['role']}")
                
                structured_str = ", ".join(structured_info) if structured_info else ""
                detail_line = f"{i}. [{mem.namespace[2]}] {summary}"
                if structured_str:
                    detail_line += f"\n   Structured Data: {structured_str}"
                detail_line += f"\n   Details: {text}\n   Date: {created}"
                memory_details.append(detail_line)
            
            memory_context = "\n\n".join(memory_details)
            
            # Use LLM to generate response from memories
            recall_prompt = f"""You are a helpful assistant with access to past conversation memories. 
Answer the user's question using ONLY the information from these memories.

User Question: {user_query}

Relevant Memories:
{memory_context}

Provide a natural, conversational response. If the memories contain the answer, use it confidently. 
If not, politely say you don't have that information in your memory yet."""
            
            response = self.model.invoke(recall_prompt).content
        else:
            response = "I don't have any relevant memories about that yet. Could you tell me more?"
        
        execution_times = state.get("execution_times", {})
        execution_times[f"{self.name}_recall"] = time.time() - start_time
        
        result = {
            "memory_response": response,
            "execution_times": execution_times
        }
        self._log(f"_recall_memory returning: memory_response={response[:100]}, execution_times keys={list(execution_times.keys())}")
        return result
    
    def create_recall_wrapper(self, store: BaseStore):
        """
        Create a wrapper for _recall_memory that has access to store.
        
        This is needed because LangGraph nodes can access store via RunnableConfig,
        but we need to pass it explicitly here.
        
        Args:
            store: Memory store instance
            
        Returns:
            Wrapped recall function
        """
        def _recall_with_store(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
            import time
            start_time = time.time()
            
            user_id = config["configurable"]["user_id"]
            user_query = state.get("user_query", "")
            if not user_query:
                return {
                    "memory_response": "I don't have any query to search for in memory.",
                    "execution_times": {f"{self.name}_recall": time.time() - start_time}
                }
            
            # Search across all memory categories
            all_memories = []
            for category in ["personal_info", "marketing_insights", "optimization", 
                           "inference", "data_insights", "miscellaneous"]:
                namespace = ("user", user_id, category)
                memories = store.search(namespace, query=user_query, limit=3)
                all_memories.extend(memories)
            
            # Format memories for LLM
            if all_memories:
                memory_details = []
                for i, mem in enumerate(all_memories[:5], 1):
                    summary = mem.value.get('summary', '')
                    text = mem.value.get('text', '')
                    created = mem.value.get('created_at', '')
                    memory_details.append(f"{i}. [{mem.namespace[2]}] {summary}\n   Details: {text}\n   Date: {created}")
                
                memory_context = "\n\n".join(memory_details)
                
                # Use LLM to generate response from memories
                recall_prompt = f"""You are a helpful assistant with access to past conversation memories. 
Answer the user's question using ONLY the information from these memories.

User Question: {user_query}

Relevant Memories:
{memory_context}

Provide a natural, conversational response. If the memories contain the answer, use it confidently. 
If not, politely say you don't have that information in your memory yet."""
                
                response = self.model.invoke(recall_prompt).content
            else:
                response = "I don't have any relevant memories about that yet. Could you tell me more?"
            
            execution_times = state.get("execution_times", {})
            execution_times[f"{self.name}_recall"] = time.time() - start_time
            
            return {
                "memory_response": response,
                "execution_times": execution_times
            }
        
        return _recall_with_store
