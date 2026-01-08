# Autonomous Agents Implementation

This folder contains an alternative implementation of the multi-agent system using **autonomous agents** with internal sub-graphs, as opposed to the orchestrated pattern in the parent directory.

## Architecture Comparison

### Orchestrated Pattern (Parent Directory)
```
Supervisor → Routes → Simple Function Nodes
                      ├─ genie_node()
                      ├─ search_node()
                      └─ memory_node()
```
- **Centralized routing**: Supervisor makes all decisions
- **Simple nodes**: Functions that execute one task
- **Direct tool calls**: Tools invoked directly, no LLM decision
- **Fast & predictable**: One path, minimal latency
- **Lower cost**: Fewer LLM calls

### Autonomous Pattern (This Directory)
```
Supervisor → Routes → Autonomous Agent Classes
                      ├─ AutonomousGenieAgent (sub-graph)
                      ├─ AutonomousSearchAgent (sub-graph)
                      └─ AutonomousMemoryAgent (sub-graph)
```
- **Distributed routing**: Each agent makes internal decisions
- **Complex agents**: Classes with compiled StateGraphs
- **Tool binding**: LLM decides when/how to use tools
- **Flexible & adaptive**: Multiple internal paths
- **Higher cost**: More LLM calls per agent

## Files

### `base_agent.py`
Base class for all autonomous agents with common infrastructure:
- Entry/exit node wrappers with timing
- Tool execution pattern
- Logging and error handling
- Helper methods (extract_user_query, track_execution)

### `search_agent.py`
**AutonomousSearchAgent**: Web search with internal decision-making
- **Internal workflow**: entry → plan → execute_tools → synthesize → exit
- **Decisions**: Whether to search, what query to use, how to format results
- **Benefits**: Can skip search if unnecessary, adapts query based on context

### `memory_agent.py`
**AutonomousMemoryAgent**: Memory with intent classification and hierarchical storage
- **Internal workflow**: entry → classify → store/recall → exit
- **Decisions**: Is user sharing info or asking for recall?
- **Benefits**: Intelligent classification, adaptive response style, category-based organization

#### Memory Implementation: Custom vs LangMem

**Our Custom Approach (Current Implementation)**
```python
# Hierarchical namespace with categories
namespace = ("user", user_id, category)
# e.g., ("user", "alice-123", "personal_info")
#       ("user", "alice-123", "marketing_insights")

# Manual classification and recall
_classify_intent() → LLM decides store vs recall
_recall_memory() → Search + LLM-generated response
```

**Advantages Over LangMem Tools:**

✅ **Domain-Specific Categories**: Six business-relevant categories (personal_info, marketing_insights, optimization, inference, data_insights, miscellaneous)
  - LangMem: Generic memory storage without built-in categorization
  - Our approach: Organized by domain, easier to query specific memory types

✅ **Custom Classification Logic**: Manual store vs recall classification with business context
  - LangMem: Relies on tool calling based on generic instructions
  - Our approach: Tailored prompts for marketing/data context

✅ **Flexible Response Generation**: LLM-generated responses from multiple memory sources
  - LangMem: Returns raw search results or structured memory objects
  - Our approach: Synthesizes information from up to 5 memories across categories into natural responses

✅ **Cross-Category Search**: Searches all categories simultaneously for comprehensive recall
  - LangMem: Single namespace search per tool call
  - Our approach: Parallel search across all 6 categories, ranked by relevance

✅ **Full Control Over Storage Schema**: Custom fields (summary, text, category, human_message, ai_message, created_at)
  - LangMem: Fixed schema unless custom Pydantic models provided
  - Our approach: Schema optimized for conversation summarization and retrieval

✅ **Better Integration with Autonomous Pattern**: Fits naturally into sub-graph workflow
  - LangMem: Designed for tool-calling agents (ReAct pattern)
  - Our approach: Node-based routing with explicit state management

**When LangMem Might Be Better:**
- ❌ Simpler requirements (generic memory, no categories)
- ❌ Using `create_react_agent` with tool calling
- ❌ Want built-in CRUD operations (create, update, delete by ID)
- ❌ Prefer less custom code to maintain

**Our Choice**: The added complexity of custom classification and categorization provides **significant value** for business context understanding and organized retrieval, outweighing the simplicity benefits of LangMem's generic tools.

### `genie_agent.py`
**AutonomousGenieAgent**: Databricks Genie with validation and HITL
- **Internal workflow**: entry → validate → execute → review → exit
- **Decisions**: Is query appropriate? Does it need human review?
- **Benefits**: Validates before expensive API calls, implements HITL for sensitive ops

### `graph.py`
Main workflow using autonomous agents as compiled sub-graphs.
- Same supervisor routing as orchestrated pattern
- Agents are added as `agent.create_agent()` (returns CompiledStateGraph)
- Each agent runs its own internal graph when invoked

## Usage

### To switch from orchestrated to autonomous pattern:

**Option 1: Update app.py import**
```python
# Change this:
from graph import get_graph

# To this:
from autonomous.graph import get_graph
```

**Option 2: Test side-by-side**
```python
# Run orchestrated version
from graph import get_graph as get_orchestrated_graph
orchestrated = get_orchestrated_graph()

# Run autonomous version
from autonomous.graph import get_graph as get_autonomous_graph
autonomous = get_autonomous_graph()
```

## When to Use Each Pattern

### Use Orchestrated (Parent Directory) When:
- ✅ Speed and cost are critical
- ✅ Predictable behavior is important
- ✅ Simple workflows (one task per agent)
- ✅ Debugging ease is valued
- ✅ **Production recommendation for most use cases**

### Use Autonomous (This Directory) When:
- ✅ Agents need complex multi-step workflows
- ✅ Agents should adapt to context dynamically
- ✅ Tool usage decisions should be intelligent
- ✅ Flexibility is more important than speed
- ✅ **Research/experimentation scenarios**

## Key Differences in Agent Behavior

### Search Agent

**Orchestrated**: Always searches when routed
```python
def search_node(state):
    results = tavily_tool.invoke(query)  # Always searches
    return {"messages": [results]}
```

**Autonomous**: Decides whether to search
```python
class AutonomousSearchAgent:
    def _plan(self, state):
        # LLM decides: "Should I search or clarify?"
        if query_is_vague:
            return "I need more specific information"
        else:
            return call_tavily()
```

### Memory Agent

**Orchestrated**: Simple recall function
```python
def recall_memory_node(state, store):
    memories = store.search(query)
    return {"messages": [format_memories(memories)]}
```

**Autonomous**: Classifies intent first
```python
class AutonomousMemoryAgent:
    def _classify(self, state):
        # LLM decides: "Is user sharing or asking?"
        if sharing_info:
            route_to_store()
        else:
            route_to_recall()
```

## Performance Comparison

| Metric | Orchestrated | Autonomous |
|--------|-------------|-----------|
| **Latency** | ~2-3 seconds | ~4-6 seconds |
| **Cost per query** | 1x baseline | 2-3x baseline |
| **LLM calls** | 2-3 | 5-8 |
| **Flexibility** | Low | High |
| **Predictability** | High | Medium |

## Testing

Both patterns share the same state definition (`AgentState`) and can be tested with identical inputs:

```python
config = {"configurable": {"user_id": "test_user"}}
state = {"messages": [HumanMessage(content="What's the weather?")]}

# Test both
orchestrated_result = orchestrated_graph.invoke(state, config)
autonomous_result = autonomous_graph.invoke(state, config)
```

## Migration Path

1. **Start with orchestrated** (proven, production-ready)
2. **Identify complex workflows** that need more flexibility
3. **Migrate specific agents** to autonomous pattern
4. **Use hybrid approach**: Keep simple agents orchestrated, complex agents autonomous
5. **Measure impact**: Compare latency, cost, and quality

## Contributing

When adding new agents:
1. Extend `AutonomousAgent` base class
2. Implement `_build_graph()`, `_entry_node()`, `_exit_node()`
3. Add agent-specific nodes with internal routing logic
4. Update `graph.py` to include new agent
5. Update `__init__.py` exports
