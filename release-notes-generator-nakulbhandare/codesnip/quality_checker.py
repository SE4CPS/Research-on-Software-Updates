import subprocess

def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def run_all_checks():
    return {
        "pytest": run_command("pytest"),
        "coverage": run_command("coverage run -m pytest && coverage report"),
        "pylint": run_command("pylint codesnip"),
        "bandit": run_command("bandit -r codesnip")
    }
