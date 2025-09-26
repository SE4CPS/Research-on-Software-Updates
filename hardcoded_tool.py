# hardcoded_tool.py
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from hardcoded_que import fetch_last_week_problem_updates, build_prompt

@tool("last_week_problem_updates", return_direct=True)
def last_week_problem_updates() -> str:
    """
    Hard-coded question:
    'Which new patch updates released last week have problems?'
    Pulls from ReleaseTrain OS + Reddit, filters last 7 days, and summarizes via Gemini.
    """
    os_hits, rd_hits, now = fetch_last_week_problem_updates()
    prompt = build_prompt(os_hits, rd_hits, now)
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)
    resp = model.invoke([{"role": "user", "content": prompt}])
    return resp.text()