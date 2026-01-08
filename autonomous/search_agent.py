"""
Autonomous Search Agent - Simplified version.
"""

from typing import Dict, Any
from langgraph.graph import StateGraph
from .base_agent import AutonomousAgent


class AutonomousSearchAgent(AutonomousAgent):
    """Simplified autonomous agent for web search."""
    
    def __init__(self, model, tavily_tool):
        """Initialize search agent."""
        super().__init__(
            name="Search",
            model=model,
            tools=[tavily_tool],
            verbose=True
        )
    
    def _build_graph(self, graph: StateGraph) -> StateGraph:
        """Define workflow: entry -> execute -> exit"""
        graph.add_node("execute", self._execute_search)
        graph.add_edge("entry", "execute")
        graph.add_edge("execute", "exit")
        return graph
    
    def _entry_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Entry node - just pass through"""
        return {}
    
    def _execute_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search using Tavily tool"""
        self._log("Executing search")
        
        query = self.extract_user_query(state)
        
        try:
            # Use the tool directly
            if not self.tools:
                raise ValueError("No search tool configured")
            search_results = self.tools[0].invoke(query)
            self._log(f"Search returned {len(search_results)} results")
            
            # Synthesize results into one answer
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Combine search results
            combined_results = "\n\n".join([
                f"Result {i+1}: {r.get('content', '')[:400]}\nURL: {r.get('url', 'N/A')}"
                for i, r in enumerate(search_results[:3])
            ])
            
            # Get unique URLs
            sources = [r.get('url') for r in search_results[:3] if r.get('url')]
            
            synthesis_prompt = f"""Based on these search results, provide a single concise answer to the query.

Query: {query}

Search Results:
{combined_results}

Provide a clear, consolidated answer (2-3 sentences). Don't repeat the same information."""
            
            response = self.model.invoke([
                SystemMessage(content="Synthesize search results into a concise answer."),
                HumanMessage(content=synthesis_prompt)
            ])
            
            # Format with sources
            formatted = response.content
            if sources:
                formatted += "\n\n**Sources:**\n" + "\n".join([f"• [{url}]({url})" for url in sources[:3]])
            
            return {
                "messages": [{
                    "role": "assistant",
                    "name": self.name,
                    "content": formatted
                }]
            }
        except Exception as e:
            self._log(f"Search failed: {e}", level="error")
            return {
                "messages": [{
                    "role": "assistant",
                    "name": self.name,
                    "content": f"Search failed: {str(e)}"
                }]
            }
    
    def _exit_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Exit node - messages already set"""
        return {}
