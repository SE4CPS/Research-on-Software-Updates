# import logging
# import time
# from openai import OpenAI, RateLimitError

# logger = logging.getLogger(__name__)

# def get_best_model(client):
#     """
#     Determine the best available model from the API, fallback gracefully if necessary.
#     """
#     try:
#         models = client.models.list()
#         model_ids = [model.id for model in models.data]
#         if "gpt-4o-mini" in model_ids:
#             return "gpt-4o-mini"
#         elif "gpt-4o" in model_ids:
#             return "gpt-4o"
#         else:
#             return "gpt-3.5-turbo"
#     except Exception as e:
#         logger.warning(f"Could not fetch model list, defaulting to gpt-3.5-turbo: {e}")
#         return "gpt-3.5-turbo"

# def generate_release_notes(pr_data, quality_reports, api_key, code_diff,debug=False):

#     if debug:
#         logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#         logger.debug("Debug mode enabled: Detailed logging active")
#     else:
#         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


#     """
#     Generate production-ready release notes using OpenAI chat completions.
#     Implements retry with exponential backoff for rate limiting.
#     """
#     logger.info("Initializing OpenAI client")
#     client = OpenAI(api_key=api_key)
#     model = get_best_model(client)

#     prompt = f"""
# You are an expert AI DevOps assistant 🤖. Your job is to generate clean, well-structured, production-ready release notes 📝.

# Format:
# - Add section headers (like `## Features`, `## Bug Fixes`, `## Code Quality`, etc.)
# - Use emojis to highlight the section purpose.
# - Be concise but informative.
# - Use bullet points (✅, 🐛, 📈, ⚠️) and maintain professional tone.

# ---

# ### 🚀 Pull Request Summary
# - PR #{pr_data['number']}: {pr_data['title']}
# {pr_data['body']}

# ### 📊 Quality Reports
# """
#     for tool, result in quality_reports.items():
#         prompt += f"\n#### 🛠️ {tool.upper()}\n{result}\n"

#     prompt += f"\n### 🧾 Code Diff\n{code_diff}\n"

#     max_retries = 3
#     backoff = 1  # seconds

#     for attempt in range(max_retries):
#         try:
#             logger.debug(f"Sending prompt to OpenAI (attempt {attempt+1}) with model: {model}")
#             response = client.chat.completions.create(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0.2,
#             )
#             logger.info(f"OpenAI API call successful on attempt {attempt+1}")
#             content = response.choices[0].message.content
#             logger.debug(f"Received content (truncated): {content[:1000]}")
#             return content

#         except RateLimitError as e:
#             logger.warning(f"Rate limit error on attempt {attempt+1}/{max_retries}: {e}")
#             if attempt < max_retries - 1:
#                 logger.info(f"Retrying after {backoff} seconds...")
#                 time.sleep(backoff)
#                 backoff *= 2  # exponential backoff
#             else:
#                 logger.error("Max retries reached due to rate limits. Raising exception.")
#                 raise

#         except Exception as e:
#             logger.error(f"OpenAI API call failed due to unexpected error: {e}")
#             raise


import logging
import time
from openai import OpenAI, RateLimitError

logger = logging.getLogger(__name__)

def get_best_model(client):
    try:
        models = client.models.list()
        model_ids = [model.id for model in models.data]
        if "gpt-4o-mini" in model_ids:
            return "gpt-4o-mini"
        elif "gpt-4o" in model_ids:
            return "gpt-4o"
        else:
            return "gpt-3.5-turbo"
    except Exception as e:
        logger.warning(f"Could not fetch model list, defaulting to gpt-3.5-turbo: {e}")
        return "gpt-3.5-turbo"

def generate_release_notes(pr_data, quality_reports, api_key, code_diff, code_issues, system_metrics, debug=False):
    if debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        logger.debug("Debug mode enabled: Detailed logging active")
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger.info("Initializing OpenAI client")
    client = OpenAI(api_key=api_key)
    model = get_best_model(client)

    prompt = f"""
You are an expert AI DevOps assistant 🤖. Your job is to generate clean, well-structured, production-ready release notes 📝.

Format:
- Add section headers (like `## Features`, `## Bug Fixes`, `## Code Quality`, etc.)
- Use emojis to highlight the section purpose.
- Be concise but informative.
- Use bullet points (✅, 🐛, 📈, ⚠️) and maintain professional tone.

---

### 🚀 Pull Request Summary
- PR #{pr_data['number']}: {pr_data['title']}
{pr_data['body']}

### 📊 Quality Reports
"""
    for tool, result in quality_reports.items():
        prompt += f"\n#### 🛠️ {tool.upper()}\n{result}\n"

    prompt += f"""
### 🧠 System Resource Usage
- CPU usage during analysis: {system_metrics['cpu_usage_percent']}%
- Memory before tests: {system_metrics['memory_before']}%
- Memory after tests: {system_metrics['memory_after']}%

### 🔎 Code Line Issues
"""
    if code_issues:
        for file, issues in code_issues.items():
            prompt += f"\n#### 📂 {file}\n"
            for issue in issues:
                prompt += f"- ⚠️ {issue}\n"
    else:
        prompt += "\n✅ No major issues detected in changed lines.\n"

    prompt += f"\n### 🧾 Code Diff\n{code_diff}\n"

    max_retries = 3
    backoff = 1

    for attempt in range(max_retries):
        try:
            logger.debug(f"Sending prompt to OpenAI (attempt {attempt+1}) with model: {model}")
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            logger.info(f"OpenAI API call successful on attempt {attempt+1}")
            content = response.choices[0].message.content
            logger.debug(f"Received content (truncated): {content[:1000]}")
            return content

        except RateLimitError as e:
            logger.warning(f"Rate limit error on attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying after {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
            else:
                logger.error("Max retries reached due to rate limits. Raising exception.")
                raise

        except Exception as e:
            logger.error(f"OpenAI API call failed due to unexpected error: {e}")
            raise
