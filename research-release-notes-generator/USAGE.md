# Codesnip Usage Guide

## Table of Contents
- [Getting Started](#getting-started)
- [Authentication Setup](#authentication-setup)
- [Command Reference](#command-reference)
- [Usage Examples](#usage-examples)
- [Output Format](#output-format)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

Codesnip is a powerful CLI tool for analyzing GitHub Pull Requests and generating AI-powered release notes with comprehensive code quality checks.

### Prerequisites
Before using Codesnip, ensure you have:
1. Python 3.8+ installed
2. Codesnip package installed (`pip install codesnip`)
3. GitHub Personal Access Token
4. OpenAI API Key
5. Valgrind (optional, for memory leak detection)

## Authentication Setup

### GitHub Token Setup
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate a new token with the following scopes:
   - `repo` (for private repositories)
   - `public_repo` (for public repositories)
3. Copy the token for use with `--token` parameter

### OpenAI API Key Setup
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key for use with `--openai-key` parameter

### Environment Variables (Recommended)
For security, set up environment variables:
```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
```

Then use in commands:
```bash
codesnip analyze --token "$GITHUB_TOKEN" --openai-key "$OPENAI_API_KEY" ...
```

## Command Reference

### Global Options
```
--debug    Enable debug logging for detailed output
--help     Show help message and exit
```

### Commands Overview
- `analyze` - Full PR analysis with quality checks and AI release notes
- `fetch` - Retrieve PR data without running quality checks

## Usage Examples

### Basic Analysis
Analyze a Pull Request and generate release notes:
```bash
codesnip analyze \
  --pr 123 \
  --repo "myorg/myproject" \
  --token "ghp_xxxxxxxxxxxx" \
  --openai-key "sk-xxxxxxxxxxxx"
```

### Custom Output File
Specify a custom output file for release notes:
```bash
codesnip analyze \
  --pr 456 \
  --repo "company/backend-api" \
  --token "$GITHUB_TOKEN" \
  --openai-key "$OPENAI_API_KEY" \
  --output "release-v2.3.1.md"
```

### Debug Mode
Enable debug logging for troubleshooting:
```bash
codesnip --debug analyze \
  --pr 789 \
  --repo "team/frontend-app" \
  --token "$GITHUB_TOKEN" \
  --openai-key "$OPENAI_API_KEY"
```

### Fetch PR Data Only
Retrieve PR information without quality analysis:
```bash
codesnip fetch \
  --pr 101 \
  --repo "open-source/library" \
  --token "$GITHUB_TOKEN" \
  --openai-key "$OPENAI_API_KEY"
```

### Batch Processing Example
Process multiple PRs using a shell script:
```bash
#!/bin/bash
REPO="myorg/myproject"
PRS=(123 124 125 126)

for pr in "${PRS[@]}"; do
  echo "Processing PR #$pr"
  codesnip analyze \
    --pr "$pr" \
    --repo "$REPO" \
    --token "$GITHUB_TOKEN" \
    --openai-key "$OPENAI_API_KEY" \
    --output "release-notes-pr-${pr}.md"
done
```

## Output Format

### Release Notes Structure
Codesnip generates structured markdown files with the following sections:

```markdown
# 📅 Release Notes - Version X.X.X

## 🚀 Features
- New functionality and enhancements

## 🐛 Bug Fixes
- Fixed issues and patches

## ⚠️ Code Quality
- Pylint and code style issues
- Security recommendations from Bandit

## 📈 Quality Reports
- Test execution results
- Coverage analysis
- Security scan results

## 🧠 System Resource Usage
- CPU usage during analysis: XX.X%
- Memory usage before tests: XX.X%
- Memory usage after tests: XX.X%

## 🔍 Code Line Issues
- File-specific code quality issues
- Line length violations
- Unsafe code patterns

## 🧾 Code Diff
- Complete diff of changes
```

### Console Output
During execution, you'll see:
```
2024-01-15 10:30:15 - INFO - Starting analysis for PR #123 in repo myorg/myproject
2024-01-15 10:30:16 - INFO - Fetching PR data from URL: https://api.github.com/repos/myorg/myproject/pulls/123
2024-01-15 10:30:17 - INFO - Running quality checks...
2024-01-15 10:30:20 - INFO - Generating release notes with AI model
2024-01-15 10:30:25 - INFO - Release notes written to release-notes.md
Release notes written to release-notes.md
```

## Best Practices

### Security
1. **Never expose API keys** in command history or scripts
2. **Use environment variables** for sensitive credentials
3. **Rotate tokens regularly** for security compliance
4. **Review generated content** before publishing release notes

### Performance
1. **Run on dedicated systems** for resource-intensive analysis
2. **Monitor system resources** during large PR analysis
3. **Use debug mode sparingly** to avoid excessive logging
4. **Consider timeout implications** for large repositories

### Quality Analysis
1. **Ensure test suite exists** before running coverage analysis
2. **Install pylint configuration** matching your project standards
3. **Review security recommendations** from Bandit scans
4. **Address code quality issues** identified in reports

### Release Notes
1. **Review AI-generated content** for accuracy and tone
2. **Customize output filenames** for version tracking
3. **Include context** in PR descriptions for better AI analysis
4. **Maintain consistent formatting** across releases

## Troubleshooting

### Common Issues

#### Authentication Errors
```
Error: GitHub API responded with status code: 401
```
**Solution**: Verify your GitHub token has correct permissions and hasn't expired.

#### OpenAI API Errors
```
Error: OpenAI API call failed due to unexpected error
```
**Solutions**:
- Check API key validity and billing status
- Verify internet connectivity
- Review rate limits and usage quotas

#### Memory Leak Detection Timeout
```
Memory leak check timed out.
```
**Solution**: This is normal on systems without Valgrind or with complex test suites. The tool continues with other checks.

#### No Tests Found
```
PYTEST: No tests were executed.
```
**Solution**: Ensure your project has test files and pytest is properly configured.

#### Large PR Analysis
```
Response too large or timeout errors
```
**Solutions**:
- Use `fetch` command for large PRs to get basic info
- Split large PRs into smaller ones
- Increase system resources

### Debug Mode Output
Enable debug mode to see detailed execution:
```bash
codesnip --debug analyze --pr 123 --repo "org/repo" --token "$TOKEN" --openai-key "$KEY"
```

Debug output includes:
- Detailed API request/response information
- Step-by-step execution progress
- Resource usage monitoring
- Error stack traces

### Getting Help
1. Use `codesnip --help` for command overview
2. Use `codesnip analyze --help` for specific command help
3. Enable debug mode for detailed error information
4. Check system resources and dependencies

### Performance Optimization
- **Smaller PRs**: Analyze smaller, focused PRs for faster processing
- **Resource monitoring**: Ensure adequate CPU and memory availability
- **Network stability**: Stable internet connection for API calls
- **Dependency management**: Keep dependencies updated

## Advanced Usage

### Integration with CI/CD
```yaml
# GitHub Actions example
name: PR Analysis
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install Codesnip
        run: pip install codesnip
      - name: Analyze PR
        run: |
          codesnip analyze \
            --pr ${{ github.event.number }} \
            --repo ${{ github.repository }} \
            --token ${{ secrets.GITHUB_TOKEN }} \
            --openai-key ${{ secrets.OPENAI_API_KEY }}
```

### Custom Scripts
Create reusable scripts for your workflow:
```bash
#!/bin/bash
# analyze-pr.sh
set -e

PR_NUMBER="$1"
REPO="$2"
OUTPUT_DIR="./release-notes"

if [ -z "$PR_NUMBER" ] || [ -z "$REPO" ]; then
    echo "Usage: $0 <pr_number> <repo>"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

codesnip analyze \
    --pr "$PR_NUMBER" \
    --repo "$REPO" \
    --token "$GITHUB_TOKEN" \
    --openai-key "$OPENAI_API_KEY" \
    --output "$OUTPUT_DIR/release-pr-${PR_NUMBER}.md"

echo "Analysis complete: $OUTPUT_DIR/release-pr-${PR_NUMBER}.md"
```

This comprehensive usage guide covers all aspects of using Codesnip effectively and securely.