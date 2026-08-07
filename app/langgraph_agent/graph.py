from langgraph.graph import StateGraph
from langgraph.graph import END

from app.langgraph_agent.state import AgentState
from app.langgraph_agent.nodes import (
    reflection_node,
    rewrite_node,
    rag_node,
)

from app.langgraph_agent.edges import should_rewrite


builder = StateGraph(AgentState)

builder.add_node(
    "reflection",
    reflection_node,
)

builder.add_node(
    "rewrite",
    rewrite_node,
)

builder.add_node(
    "rag",
    rag_node,
)

builder.set_entry_point("reflection")

builder.add_conditional_edges(
    "reflection",
    should_rewrite,
    {
        "rewrite": "rewrite",
        "rag": "rag",
    },
)

builder.add_edge(
    "rewrite",
    "rag",
)

builder.add_edge(
    "rag",
    END,
)

graph = builder.compile()