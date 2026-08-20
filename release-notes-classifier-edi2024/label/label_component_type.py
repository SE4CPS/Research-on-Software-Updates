"""
classify_component_type.py

Keyword-based classifier for "Component Type" (see Berhe et al., "Triage
Software Update Impact via Release Notes Classification," Procedia Computer
Science, 2024).

Pure classification logic, no external dependencies. Pass any release-note
text to classify_text() and get back the matching component category labels.
"""

import re

# Define Categories for Classification
categories_component = {
    "gpt": ["GPT", "GPT model", "OpenAI", "LLM", "GPT-2", "GPT-3", "GPT-4", "NLP", "ChatGPT", "Transformer", "Language Model", "AI"],
    "bitcoin": ["bitcoin", "ethereum", "crypto", "cryptocurrency", "blockchain", "cryptographic", "mining", "wallet", "defi", "altcoin", "NFT"],
    "browser": ["browser", "chrome", "firefox", "edge", "safari", "opera", "chromium", "brave", "vivaldi", "web browser", "extensions", "bookmark"],
    "database": ["database", "sql", "nosql", "key-value store", "mongodb", "postgresql", "cassandra", "mysql", "rdbms", "dbms", "sqlite"],
    "os": ["os", "ios", "android", "red hat", "windows", "linux", "ubuntu", "macos", "centos", "kernel", "fedora", "linux-dist", "apple", "phone", "operating system", "wayland", "grub", "boot", "nvidia"],
    "server": ["server", "jenkins", "nginx", "apache", "tomcat", "web server", "application server", "http server", "staging environment"],
    "ide": ["editor", "ide", "vim", "vscode", "pycharm", "eclipse", "command line", "shell", "command", "editor", "integrated development environment"],
    "api": ["api", "swagger", "rest", "graphql", "soap", "application programming interface", "endpoint", "api gateway", "websockets"],
    "framework": ["pytorch", "framework", "react", "vue", "angular", "django", "spring", "rails", "laravel", "frameworks", "asp", "ktor"],
    "library": ["library", "jquery", "react router", "redux", "library", "toolkit", "libs", "lib"],
    "language": [
        "language", "programming language", "coding language", "scripting language",
        "python", "python2", "python3", "javascript", "typescript", "java", "c#", "c++", "c",
        "rust", "go", "ruby", "swift", "kotlin", "scala", "perl", "r", "php", "dart", "lua",
        "haskell", "clojure", "elixir", "f#", "objective-c", "visual basic", ".net", "shell",
        "bash", "zsh", "powershell", "sql", "pl/sql", "html", "css", "jsx", "tsx", "fortran",
        "cobol", "assembly", "lisp", "scheme", "prolog", "matlab", "groovy", "julia", "sas"
    ],
    "versioning": ["versioning", "version management", "git", "svn", "mercurial", "version control", "github", "branching", "merging", "release notes"],
    "app": ["app", "wordpress", "drupal", "joomla", "application", "web app", "mobile app", "game", "pygame"],
    "repository": ["repository", "repositories", "package", "package manager", "repo", "version control", "npm", "nuget", "maven", "pypi", "cran", "github", "gitlab", "bitbucket", "dependency", "pnpm"],
    "cloud": ["cloud", "aws", "azure", "cloud computing", "lambda", "azure functions", "app engine", "gcp", "sap"],
    "containerization": ["container", "docker", "kubernetes", "docker swarm", "openshift", "container orchestration", "containers"],
    "testing": ["testing", "unit testing", "integration testing", "test automation", "selenium", "junit", "pytest", "tdd", "test driven development"],
    "mobile": ["mobile", "mobile app", "react native", "swift", "kotlin", "ios", "android", "mobile development"],
    "web": ["web", "webpage", "website", "html", "css", "javascript", "web development", "vue.js", "node.js", "bootstrap", "web design", "frontend", "backend", "nextjs"],
    "data": ["data", "big data", "data analytics", "data science", "hadoop", "spark", "pandas", "data engineering", "data analysis", "table", "import wizard"],
    "agile": ["agile", "scrum", "kanban", "agile methodology", "jira", "trello", "scrum master", "agile development"],
    "devops": ["devops", "continuous integration", "continuous deployment", "devops practices", "jenkins pipeline", "ansible", "docker compose", "ci/cd", "infrastructure as code"],
    "firmware": ["firmware", "firmwares", "embedded", "flash"],
    "design": ["design", "ui", "ux", "user experience", "frontend", "interface", "figma", "theme", "css styling"],
    "game": ["game", "gaming", "game engine", "unity", "unreal", "godot", "roblox"],
    "machine_learning": ["machine learning", "ml", "deep learning", "neural network", "artificial intelligence", "ai", "tensorflow", "pytorch", "scikit-learn"],
    "cli": ["command line", "cli", "terminal", "bash", "shell", "zsh", "powershell", "cmd", "command prompt"]
}


def classify_text(text):
    """Return the list of Component Type category labels matched in `text`."""
    if not text:
        return []

    words = re.findall(r'\b\w+\b', text.lower())
    matches = set()

    for word in words:
        for category, keywords in categories_component.items():
            if word.lower() in (kw.lower() for kw in keywords):
                matches.add(category.upper())

    return list(matches) if matches else []


if __name__ == "__main__":
    sample = "Fixed a memory leak in the Chrome browser extension API and updated the Docker container image."
    print(classify_text(sample))
