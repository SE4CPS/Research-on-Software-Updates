import click
import logging
import requests
import psutil
import subprocess
from collections import defaultdict
from codesnip.github_fetcher import fetch_pr_data
from codesnip.quality_checker import run_all_checks
from codesnip.openai_client import generate_release_notes

# Set up logger
logger = logging.getLogger(__name__)

def configure_logging(debug):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

def fetch_pr_data_with_logs(repo, pr_number, token):
    base_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    headers = {"Authorization": f"token {token}"}
    logger.info(f"Fetching PR data from URL: {base_url}")
    resp = requests.get(base_url, headers=headers, timeout=30)
    logger.info(f"GitHub API responded with status code: {resp.status_code}")
    pr = resp.json()

    diff_url = pr.get("diff_url", "")
    logger.info(f"Fetching PR diff from URL: {diff_url}")
    # Only fetch diff if URL is valid
    if diff_url:
        diff_resp = requests.get(diff_url, headers=headers, timeout=30)
        logger.info(f"Diff fetch status code: {diff_resp.status_code}")
    else:
        logger.warning("No diff URL found in PR data")
        diff_resp = None

    pr_data = {
        "number": pr.get("number"),
        "title": pr.get("title"),
        "body": pr.get("body", ""),
        "merged_at": pr.get("merged_at", ""),
        "diff": diff_resp.text if diff_resp and diff_resp.status_code == 200 else ""
    }
    return pr_data

def analyze_code_diff_by_file(code_diff):
    logger.info("Analyzing code diff line by line...")
    file_diffs = defaultdict(list)
    current_file = None
    for line in code_diff.splitlines():
        if line.startswith("diff --git"):
            parts = line.split(" b/")
            if len(parts) > 1:
                current_file = parts[-1]
        elif current_file and line.startswith('+') and not line.startswith('+++'):
            file_diffs[current_file].append(line[1:])
    
    issues = {}
    for file, lines in file_diffs.items():
        file_issues = []
        for idx, line in enumerate(lines):
            if len(line) > 120:
                file_issues.append(f"Line {idx+1} is too long (>120 chars).")
            if "eval(" in line:
                file_issues.append(f"Line {idx+1} uses `eval()` which can be unsafe.")
            if "print(" in line:
                file_issues.append(f"Line {idx+1} has `print()`. Consider using logging.")
        if file_issues:
            issues[file] = file_issues
    return issues

@click.group()
@click.option('--debug', is_flag=True, default=False, help='Enable debug logging')
@click.pass_context
def main(ctx, debug):
    """CLI entry point."""
    ctx.ensure_object(dict)
    ctx.obj['DEBUG'] = debug
    configure_logging(debug)
    logger.debug("Debug logging enabled")

@main.command()
@click.argument('pr', type=int)
@click.option('--repo', required=True, type=str, help='GitHub repository (e.g. user/repo)')
@click.option('--token', required=True, type=str, help='GitHub token')
@click.option('--openai-key', required=True, type=str, help='OpenAI API key')
@click.option('--output', default='release-notes.md', help='Output file for release notes')
@click.pass_context
def analyze(ctx, pr, repo, token, openai_key, output):
    debug = ctx.obj['DEBUG']
    logger.info(f"Starting analysis for PR #{pr} in repo {repo}")

    pr_data = fetch_pr_data_with_logs(repo, pr, token)
    code_diff = pr_data.get("diff", "")

    if not code_diff:
        logger.error("No code diff found in PR data. Aborting analysis.")
        click.echo("No code diff found in PR data.")
        return

    logger.info("Running quality checks...")
    checks = run_all_checks()

    logger.info("Checking memory usage before tests")
    mem_before = psutil.virtual_memory()

    try:
        logger.info("Running memory leak detection (valgrind)...")
        mem_leak_output = subprocess.run(
            "valgrind --leak-check=full python -m pytest",
            shell=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        checks['memory_leaks'] = mem_leak_output.stdout + mem_leak_output.stderr
        logger.info("Memory leak check completed")
    except subprocess.TimeoutExpired:
        checks['memory_leaks'] = "Memory leak check timed out."
        logger.error("Memory leak check timed out.")
    except Exception as e:
        checks['memory_leaks'] = f"Memory leak check failed: {str(e)}"
        logger.error(f"Memory leak check failed: {str(e)}")

    logger.info("Checking memory usage after tests")
    mem_after = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)
    logger.info(f"CPU usage during analysis: {cpu_percent}%")

    logger.info("Analyzing code diff...")
    code_issues = analyze_code_diff_by_file(code_diff)

    system_metrics = {
        "cpu_usage_percent": cpu_percent,
        "memory_before": mem_before.percent,
        "memory_after": mem_after.percent,
    }

    logger.info("Generating release notes with AI model")
    notes = generate_release_notes(
        pr_data,
        checks,
        openai_key,
        code_diff,
        code_issues,
        system_metrics,
        debug=debug
    )

    with open(output, 'w') as f:
        f.write(notes)
    logger.info(f"Release notes written to {output}")
    click.echo(f'Release notes written to {output}')

@main.command()
@click.argument('pr', type=int)
@click.option('--repo', required=True, type=str, help='GitHub repository (e.g. user/repo)')
@click.option('--token', required=True, type=str, help='GitHub token')
@click.option('--openai-key', required=True, type=str, help='OpenAI API key')
@click.pass_context
def fetch(ctx, pr, repo, token, openai_key):
    debug = ctx.obj['DEBUG']
    logger.info(f"Fetching data for PR #{pr} in repo {repo}")
    pr_data = fetch_pr_data_with_logs(repo, pr, token)
    click.echo(pr_data)
    logger.info(f"Fetched PR data: {pr_data}")
    click.echo(f"Fetched PR data for #{pr} in {repo}")

if __name__ == '__main__':
    main()
