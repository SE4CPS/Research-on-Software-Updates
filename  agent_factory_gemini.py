# agent_factory_gemini.py
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from releasetrain_tools import rt_os, rt_reddit
from hardcoded_tool import last_week_problem_updates

def build_agent(with_memory: bool = True):
    model = init_chat_model(
        "gemini-2.5-flash",
        model_provider="google_genai",
        temperature=0,  # deterministic
    )

    tools = [
        last_week_problem_updates,  # your hard-coded question
        rt_os,                      # ReleaseTrain OS API
        rt_reddit,                  # ReleaseTrain Reddit API
    ]

    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=MemorySaver() if with_memory else None,
    )

    def cfg(thread_id: str = "default"):
        return {"configurable": {"thread_id": thread_id}} if with_memory else {}
    return agent, cfg