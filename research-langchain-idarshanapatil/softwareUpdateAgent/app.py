from typing import Any, Dict

from langgraph.graph import END, StateGraph

from backend.models import WorkflowState
from backend.nodes.fetchers import google_news_node, reddit_questions_node, release_train_node
from backend.nodes.formatter import formatter_node
from backend.nodes.merge import merge_node
from backend.nodes.orchestrator import orchestrator_node
from backend.nodes.prioritizer import prioritize_node


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("release_train_fetch", release_train_node)
    graph.add_node("google_news_fetch", google_news_node)
    graph.add_node("reddit_questions_fetch", reddit_questions_node)
    graph.add_node("merge", merge_node)
    graph.add_node("prioritize", prioritize_node)
    graph.add_node("format", formatter_node)

    graph.set_entry_point("orchestrator")
    # Three parallel fetch nodes after orchestrator
    graph.add_edge("orchestrator", "release_train_fetch")
    graph.add_edge("orchestrator", "google_news_fetch")
    graph.add_edge("orchestrator", "reddit_questions_fetch")
    # All three complete before merge
    graph.add_edge("release_train_fetch", "merge")
    graph.add_edge("google_news_fetch", "merge")
    graph.add_edge("reddit_questions_fetch", "merge")
    graph.add_edge("merge", "prioritize")
    graph.add_edge("prioritize", "format")
    graph.add_edge("format", END)
    return graph.compile()


def run_query(user_query: str) -> Dict[str, Any]:
    app = build_graph()
    return app.invoke({"user_query": user_query})
