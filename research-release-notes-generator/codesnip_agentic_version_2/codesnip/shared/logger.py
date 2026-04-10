"""
logger.py — Verbose step-by-step logger using plain ANSI (no rich dependency).
"""
import sys
import time

_timers: dict = {}

# ANSI colors
_C = {
    'reset':  '\033[0m',
    'bold':   '\033[1m',
    'dim':    '\033[2m',
    'cyan':   '\033[36m',
    'green':  '\033[32m',
    'yellow': '\033[33m',
    'red':    '\033[31m',
    'white':  '\033[97m',
    'blue':   '\033[34m',
}

def _c(code): return _C.get(code, '')
def _ts(): return f"{_c('dim')}{time.strftime('%H:%M:%S')}{_c('reset')}"
def _ms(key):
    if key in _timers:
        ms = (time.time() - _timers[key]) * 1000
        return f" {_c('dim')}({ms:.0f}ms){_c('reset')}"
    return ""

def _print(msg): print(msg, flush=True)

def banner(title: str, subtitle: str = "") -> None:
    w = min(len(title) + 6, 72)
    border = '━' * w
    _print(f"\n{_c('cyan')}{border}{_c('reset')}")
    _print(f"{_c('cyan')}  {_c('bold')}{title}{_c('reset')}")
    if subtitle:
        _print(f"{_c('cyan')}  {_c('dim')}{subtitle}{_c('reset')}")
    _print(f"{_c('cyan')}{border}{_c('reset')}")

def section(label: str) -> None:
    _print(f"\n{_c('cyan')}{'─'*60}{_c('reset')}")
    _print(f"{_c('bold')}{_c('cyan')} {label} {_c('reset')}")

def pr_card(repo: str, pr_number: int, meta: dict) -> None:
    _print(f"\n  {_c('bold')}{meta['title']}{_c('reset')}")
    _print(f"  {_c('dim')}PR #{pr_number} · {repo} · author: {_c('cyan')}{meta['author']}{_c('reset')}")
    _print(f"  {_c('dim')}{meta.get('url','')}{_c('reset')}\n")

def start(key: str, msg: str) -> None:
    _timers[key] = time.time()
    _print(f"{_ts()} {_c('cyan')}→{_c('reset')}  {msg}")

def ok(key: str, msg: str) -> None:
    _print(f"{_ts()} {_c('green')}✔{_c('reset')}  {msg}{_ms(key)}")

def warn(key: str, msg: str) -> None:
    _print(f"{_ts()} {_c('yellow')}⚠{_c('reset')}  {msg}{_ms(key)}")

def fail(key: str, msg: str) -> None:
    _print(f"{_ts()} {_c('red')}✘{_c('reset')}  {msg}{_ms(key)}")

def info(msg: str) -> None:
    _print(f"{_ts()} {_c('dim')}·  {msg}{_c('reset')}")

def detail(label: str, value: str, color: str = "white") -> None:
    _print(f"         {_c('dim')}{label}:{_c('reset')} {_c(color)}{value}{_c('reset')}")

def blank() -> None:
    _print("")

def check_ok(msg: str) -> None:
    _print(f"         {_c('green')}✔ {msg}{_c('reset')}")

def check_fail(msg: str) -> None:
    _print(f"         {_c('red')}✘ {msg}{_c('reset')}")

def check_warn(msg: str) -> None:
    _print(f"         {_c('yellow')}⚠ {msg}{_c('reset')}")

def check_skip(msg: str) -> None:
    _print(f"         {_c('dim')}– {msg}{_c('reset')}")

def file_header(filename: str, lang: str, added: int, removed: int) -> None:
    _print(f"\n{_ts()} {_c('bold')}{filename}{_c('reset')}  {_c('dim')}{lang}{_c('reset')}  "
           f"{_c('green')}+{added}{_c('reset')} {_c('red')}-{removed}{_c('reset')}")
