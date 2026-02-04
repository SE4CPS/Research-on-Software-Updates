# from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent


# Tool
@tool
# def get_weather(city: str) -> str:
#     """Get weather for a given city."""
#     return f"It's always sunny in {city}!"

def get_weather(city: str) -> dict:
    """Get weather for a given city."""
    return {
        "tool_message": f"Found weather for {city}",
        "weather": [{"city": city, "temp_c": 18, "condition": "sunny"}]
    }

# Ollama model
llm = ChatOllama(
    model="llama3.1",
    temperature=0
)

# Agent
agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="""
Always return ONLY valid JSON.
No prefixes.
No commentary.
"""
)

# Run
result = agent.invoke({
    "messages": [{"role": "user", "content": "what is the weather in sf"}]
})

#1) basic way to print result without extra prettyness
#print(result)

#2) way using json module
# Simple, robust pretty-printer — accepts dicts and LangChain message objects
# def pretty_print_result(r):
#     import json

#     def _maybe_parse(s):
#         if isinstance(s, (dict, list)):
#             return s
#         if not isinstance(s, str):
#             return s
#         try:
#             return json.loads(s)
#         except Exception:
#             return s

#     def _get(obj, *keys):
#         """Get value from dict or message object using a list of possible attribute/key names."""
#         if obj is None:
#             return None
#         if isinstance(obj, dict):
#             for k in keys:
#                 if k in obj:
#                     return obj[k]
#             return None
#         # message object (HumanMessage, AIMessage, etc.)
#         for k in keys:
#             if hasattr(obj, k):
#                 return getattr(obj, k)
#             # some messages keep extra data in 'additional_kwargs'
#             if hasattr(obj, 'additional_kwargs') and isinstance(getattr(obj, 'additional_kwargs'), dict):
#                 if k in obj.additional_kwargs:
#                     return obj.additional_kwargs[k]
#         return None

#     # normalize top-level
#     obj = None
#     if isinstance(r, str):
#         try:
#             obj = json.loads(r)
#         except Exception:
#             print(r); return
#     else:
#         obj = r

#     # 1) direct dict with keys
#     if isinstance(obj, dict):
#         tm = _get(obj, 'tool_message')
#         weather = _get(obj, 'weather')
#     else:
#         tm = None
#         weather = None

#     # 2) inspect messages (handles dict messages and LangChain Message objects)
#     if not (tm or weather):
#         msgs = _get(obj, 'messages') if isinstance(obj, dict) else _get(obj, 'messages')
#         if isinstance(msgs, list):
#             for m in msgs:
#                 role = _get(m, 'role', 'type')
#                 # accept tool role or any message coming from the tool
#                 if role == 'tool' or 'tool' in str(_get(m, 'tool') or '').lower():
#                     content = _get(m, 'content', 'text')
#                     content = _maybe_parse(content)
#                     if isinstance(content, dict):
#                         tm = tm or content.get('tool_message')
#                         weather = weather or content.get('weather')
#                     else:
#                         tm = tm or content
#                     if tm or weather:
#                         break

#     # 3) fallback: inspect tool_calls-like structures
#     if not (tm or weather) and isinstance(obj, dict):
#         for key in ('tool_calls', 'tool_call', 'tools'):
#             calls = obj.get(key) or []
#             if isinstance(calls, list):
#                 for call in calls:
#                     name = _get(call, 'name', 'tool')
#                     resp = _maybe_parse(_get(call, 'response', 'output', 'result'))
#                     if isinstance(resp, dict):
#                         tm = tm or resp.get('tool_message')
#                         weather = weather or resp.get('weather')
#                     else:
#                         tm = tm or resp
#                     if tm or weather:
#                         break
#                 if tm or weather:
#                     break

#     # Print results (compact)
#     if tm or weather:
#         if tm:
#             print('tool_message:\n', json.dumps(_maybe_parse(tm), indent=2, ensure_ascii=False))
#         if weather:
#             print('weather array:\n', json.dumps(_maybe_parse(weather), indent=2, ensure_ascii=False))
#         return

#     # final fallback: pretty-print entire object
#     try:
#         print(json.dumps(obj, indent=2, ensure_ascii=False))
#     except Exception:
#         print(obj)

# pretty_print_result(result)

#3) way by using langchain message objects
def get_final_ai_content(result):
    from langchain_core.messages import AIMessage
    
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage) and m.content:  # this is called type safe filtering
            return m.content.replace("<|python_tag|>", "")
        
print(get_final_ai_content(result))


