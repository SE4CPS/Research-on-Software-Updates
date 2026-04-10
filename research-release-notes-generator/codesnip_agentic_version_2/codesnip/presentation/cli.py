"""
cli.py — Click CLI (replaces typer; no rich dependency required).
"""
import sys
import click

from codesnip.application.use_cases import AnalyzePRUseCase, ReleaseNotesUseCase
from codesnip.infrastructure.config_service import (
    set_key, get_all, get_ollama_url, get_ollama_model,
)
from codesnip.shared import logger

C = {
    'reset': '\033[0m', 'bold': '\033[1m', 'dim': '\033[2m',
    'cyan': '\033[36m', 'green': '\033[32m', 'yellow': '\033[33m',
    'red': '\033[31m',
}
def c(code, text): return f"{C.get(code,'')}{text}{C['reset']}"


@click.group()
def app():
    """Codesnip — AI-powered PR analyzer & release notes (local Ollama)"""
    pass


@app.command()
@click.argument('key')
@click.argument('value')
def config(key, value):
    """Set a config value (stored in ~/.codesnip/config.json).

    \b
    Keys: github_token | ollama_url | ollama_model
    """
    set_key(key, value)
    print(f"{c('green','✔')} Saved {c('bold',key)} = {c('cyan',value)}")


@app.command(name='config-show')
def config_show():
    """Show all active config values."""
    data = get_all()
    rows = [
        ('ollama_url',   data.get('ollama_url')   or get_ollama_url(),   'default' if not data.get('ollama_url')   else 'saved'),
        ('ollama_model', data.get('ollama_model') or get_ollama_model(), 'default' if not data.get('ollama_model') else 'saved'),
    ]
    token = data.get('github_token')
    rows.append(('github_token', (token[:8]+'…') if token else c('yellow','not set (public repos only)'), 'saved' if token else 'default'))

    print(f"\n{c('bold','Codesnip Config')}")
    print('─' * 48)
    for key, val, src in rows:
        print(f"  {c('cyan', f'{key:<16}')} {val}  {c('dim', src)}")
    print()


@app.command()
@click.argument('repo')
@click.argument('pr', type=int)
def analyze(repo, pr):
    """Analyse a GitHub PR — 8 categories + static checks + risk score.

    \b
    REPO  owner/repo  e.g. octocat/hello-world
    PR    pull request number
    """
    logger.banner(f"Codesnip  ·  PR #{pr}  ·  {repo}", "Fetch → Static Analysis → LLM Review")
    try:
        result = AnalyzePRUseCase().execute(repo, pr)
    except ValueError as e:
        print(f"\n{c('red', '✘ Error:')} {e}\n", file=sys.stderr)
        sys.exit(1)

    print()
    print(c('cyan', '─' * 60))
    print(c('bold', ' ANALYSIS RESULT '))
    print(c('cyan', '─' * 60))
    print()
    print(result)
    print()
    print(c('dim', '─' * 60))
    print(c('dim', 'codesnip · powered by Ollama'))


@app.command(name='release-notes')
@click.argument('repo')
@click.argument('pr', type=int)
def release_notes(repo, pr):
    """Generate structured release notes for a GitHub PR.

    \b
    REPO  owner/repo  e.g. octocat/hello-world
    PR    pull request number
    """
    logger.banner(f"Codesnip  ·  Release Notes  ·  PR #{pr}  ·  {repo}", "Fetch → Static Analysis → LLM Release Notes")
    try:
        result = ReleaseNotesUseCase().execute(repo, pr)
    except ValueError as e:
        print(f"\n{c('red', '✘ Error:')} {e}\n", file=sys.stderr)
        sys.exit(1)

    print()
    print(c('cyan', '─' * 60))
    print(c('bold', ' RELEASE NOTES '))
    print(c('cyan', '─' * 60))
    print()
    print(result)
    print()
    print(c('dim', '─' * 60))
    print(c('dim', 'codesnip · powered by Ollama'))


if __name__ == '__main__':
    app()
