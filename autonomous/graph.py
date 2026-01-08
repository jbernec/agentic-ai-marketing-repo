"""
LangGraph workflow using autonomous agents.

This is an alternative implementation to the orchestrated pattern in ../graph.py.
Each agent has its own compiled sub-graph with internal decision-making.

Key differences from orchestrated pattern:
- Agents are classes with create_agent() method returning CompiledStateGraph
- Each agent makes internal decisions about tool usage and routing
- More flexible but higher latency and cost
- Better for complex multi-step workflows within agents

To use this graph instead of the orchestrated one, update app.py:
    from autonomous.graph import get_graph
"""

import time
import uuid
from typing import Annotated, Literal
from operator import add
from datetime import datetime, timezone

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages.utils import count_tokens_approximately
from langmem.short_term import SummarizationNode, RunningSummary
from pydantic import BaseModel, Field, Field

# Import parent modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
import agents as parent_agents
import memory_setup

# Import autonomous agents
from .search_agent import AutonomousSearchAgent
from .memory_agent import AutonomousMemoryAgent
from .genie_agent import AutonomousGenieAgent
from .hitl_utils import requires_review


# Helper function to merge dictionaries
def merge_dicts(existing: dict, new: dict) -> dict:
    """Merge dictionaries, new values override existing ones."""
    return {**existing, **new}


# State definition (same as orchestrated pattern)
class AgentState(MessagesState):
    next: Literal["Genie", "Search", "Memory", "FINISH"]
    user_edits: Annotated[list[dict], add]
    execution_times: Annotated[dict[str, float], merge_dicts]
    context: dict[str, RunningSummary]
    summary: str | None
    routing_reasoning: Annotated[list[str], add]
    routing_history: list[str]
    genie_query_sql: str
    query_intent: str
    # Additional fields for autonomous agents
    user_query: str | None
    messages_to_send: list | None
    is_valid: bool | None
    genie_response: str | None
    error: str | None
    memory_response: str | None
    memory_action: str | None


class MemorySummary(BaseModel):
    category: str
    summary: str


class PersonalInfo(BaseModel):
    """Structured personal information for better retrieval"""
    name: str | None = Field(None, description="User's full name")
    location: str | None = Field(None, description="Where the user lives")
    company: str | None = Field(None, description="Where the user works")
    role: str | None = Field(None, description="User's job title or role")
    preferences: str | None = Field(None, description="User preferences as text")
    summary: str = Field(..., description="Summary of the information")

class RouteDecision(BaseModel):
    agent: Literal["Genie", "Search", "Memory"]
    reasoning: str


# Helper function to extract text from messages
def _message_to_text(message) -> str:
    if message is None:
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text_part = part.get("text")
                if text_part:
                    parts.append(text_part)
            else:
                parts.append(str(part))
        return " ".join(parts).strip()
    return str(content)


# Lazy-loaded components
def get_summarization_node():
    """Get or create summarization node"""
    model = config.get_model()
    return SummarizationNode(
        token_counter=count_tokens_approximately,
        model=model.bind(max_tokens=3000),
        max_tokens=3000,
        max_tokens_before_summary=1000,
        max_summary_tokens=2000,
    )


def get_structured_summary_model():
    """Get or create structured summary model"""
    model = config.get_model()
    return model.with_structured_output(MemorySummary)


def get_personal_info_model():
    """Get or create personal info extraction model"""
    model = config.get_model()
    return model.with_structured_output(PersonalInfo)


# Write long-term memory node (same as orchestrated)
def write_long_term_memory(state: AgentState, config: RunnableConfig, *, store):
    structured_summary_model = get_structured_summary_model()
    user_id = config["configurable"]["user_id"]

    messages_for_sum = [
        SystemMessage(
            content=(
                "You are a memory compressor. "
                "Given the following conversation, extract:\n"
                "- a high-level category from this fixed set: "
                "[\"personal_info\", \"marketing_insights\", \"optimization\", \"inference\", \"data_insights\", \"miscellaneous\"],\n"
                "- a concise summary capturing key facts, preferences, and goals.\n\n"
                "Categories guide:\n"
                "  - personal_info: Names, preferences, personal details, user information\n"
                "  - marketing_insights: Marketing strategies, campaign analysis, customer behavior\n"
                "  - optimization: Performance improvements, cost reduction, efficiency\n"
                "  - inference: Predictions, forecasts, trend analysis\n"
                "  - data_insights: Data analysis results, patterns, statistics\n"
                "  - miscellaneous: Everything else\n\n"
                "Return ONLY a structured object with fields: category, summary."
            )
        ),
        *state["messages"],
    ]

    result: MemorySummary = structured_summary_model.invoke(
        messages_for_sum,
        response_format=MemorySummary,
    )

    category = result.category
    new_summary = result.summary

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )

    human_text = _message_to_text(last_human)
    ai_text = _message_to_text(last_ai)

    text_segments = []
    if human_text:
        text_segments.append(f"Human: {human_text}")
    if ai_text:
        text_segments.append(f"AI: {ai_text}")
    memory_text = "\n".join(text_segments)

    namespace = ("user", user_id, category)

    # For personal_info category, extract structured fields
    memory_value = {
        "summary": new_summary,
        "text": memory_text,
        "category": category,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "human_message": human_text,
        "ai_message": ai_text,
    }
    
    if category == "personal_info":
        # Extract structured personal info
        extract_prompt = SystemMessage(
            content=(
                "Extract structured personal information from this conversation.\n"
                "Return fields: name, location, company, role, preferences, summary.\n"
                "Leave fields as null if not mentioned.\n\n"
                f"Conversation:\n{memory_text}\n\nSummary: {new_summary}"
            )
        )
        try:
            personal_info_model = get_personal_info_model()
            personal_info = personal_info_model.invoke([extract_prompt])
            memory_value.update({
                "name": personal_info.name,
                "location": personal_info.location,
                "company": personal_info.company,
                "role": personal_info.role,
                "preferences": personal_info.preferences,
            })
        except Exception as e:
            print(f"[WARNING] Failed to extract structured personal info: {e}")

    store.put(
        namespace,
        key=str(uuid.uuid4()),
        value=memory_value,
        index=True,
    )

    return {"messages": state["messages"], "summary": new_summary}


# Lazy-loaded autonomous agents
_search_agent = None
_memory_agent = None
_genie_agent = None


def get_search_agent():
    """Get or create autonomous search agent"""
    global _search_agent
    if _search_agent is None:
        model = config.get_model()
        tavily_tool = parent_agents.get_tavily_tool()
        _search_agent = AutonomousSearchAgent(model, tavily_tool)
    return _search_agent


def get_memory_agent():
    """Get or create autonomous memory agent"""
    global _memory_agent
    if _memory_agent is None:
        model = config.get_model()
        _memory_agent = AutonomousMemoryAgent(model)
    return _memory_agent


def get_genie_agent():
    """Get or create autonomous genie agent"""
    global _genie_agent
    if _genie_agent is None:
        model = config.get_model()
        genie = parent_agents.get_genie_agent()
        _genie_agent = AutonomousGenieAgent(model, genie)
    return _genie_agent


# Supervisor node with routing (same as orchestrated)
routing_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a routing assistant. Analyze the query and decide which agent should handle it.

Available Agents:
- Genie: SQL queries, data analysis, database questions, business intelligence
- Search: Web searches, current events, general knowledge, research
- Memory: Personal information (storing AND recalling), user preferences, personal details

IMPORTANT Routing Rules:
1. Route to Memory when user SHARES personal info ("My name is X", "I live in Y", "I work at Z")
2. Route to Memory when user ASKS about personal info ("What's my name?", "Where do I live?", "Who is my manager?")
3. Use Genie for ANY data/database/SQL questions
4. Use Search for current events, general knowledge, or web research

Choose the most appropriate agent."""),
    ("user", "{query}")
])


def get_routing_chain():
    """Get or create routing chain"""
    model = config.get_model()
    return routing_prompt | model.with_structured_output(RouteDecision)


def supervisor_node(state: AgentState, config: RunnableConfig, *, store: BaseStore):
    routing_chain = get_routing_chain()
    if state["messages"]:
        last_msg = state["messages"][-1]
        sender = getattr(last_msg, "name", None)
        
        if sender in ["Genie", "Search", "Memory"]:
            return {"next": "FINISH"}
    
    user_id = config["configurable"]["user_id"]
    user_query = [m.content for m in state["messages"] if isinstance(m, HumanMessage)][-1]
    
    # Search across all categories for relevant memories
    all_memories = []
    for category in ["personal_info", "marketing_insights", "optimization", "inference", "data_insights", "miscellaneous"]:
        namespace = ("user", user_id, category)
        memories = store.search(namespace, query=user_query, limit=2)
        all_memories.extend(memories)
    
    # Build memory context
    memory_context = "\n".join([
        f"[{m.namespace[2]}] {m.value.get('summary', '')}" 
        for m in all_memories[:3]
    ]) if all_memories else ""
    
    # Inject memory context if available
    messages_with_memory = list(state["messages"])
    if memory_context:
        memory_msg = SystemMessage(
            content=f"### Relevant Context from Past Conversations:\n{memory_context}\n\n"
                    f"Use this context to provide more personalized and contextually aware responses."
        )
        messages_with_memory.insert(-1, memory_msg)
    
    # Route using LLM
    decision = routing_chain.invoke({"query": user_query})
    
    return {
        "next": decision.agent,
        "messages": messages_with_memory
    }


def select_next_node(state: AgentState) -> str:
    return state["next"]


# Build workflow with autonomous agents
workflow = StateGraph(AgentState)
workflow.add_node("summarize", get_summarization_node())
workflow.add_node("supervisor", supervisor_node)

# Add autonomous agents as compiled sub-graphs
# Each agent.create_agent() returns a CompiledStateGraph
workflow.add_node("Genie", get_genie_agent().create_agent())
workflow.add_node("Search", get_search_agent().create_agent())
workflow.add_node("Memory", get_memory_agent().create_agent())

workflow.add_node("write_long_term_memory", write_long_term_memory)

workflow.add_edge(START, "summarize")
workflow.add_edge("summarize", "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    select_next_node,
    {"Genie": "Genie", "Search": "Search", "Memory": "Memory", "FINISH": END},
)
workflow.add_edge("Genie", "write_long_term_memory")
workflow.add_edge("Search", "write_long_term_memory")
workflow.add_edge("Memory", "write_long_term_memory")
workflow.add_edge("write_long_term_memory", "supervisor")


# Lazy-compile graph
_graph = None


def get_graph():
    """Get or create compiled graph with autonomous agents"""
    global _graph
    if _graph is None:
        checkpointer = memory_setup.get_checkpointer()
        store = memory_setup.get_store()
        _graph = workflow.compile(checkpointer=checkpointer, store=store)
    return _graph
