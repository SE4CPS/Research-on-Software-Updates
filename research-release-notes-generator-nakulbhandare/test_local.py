#!/usr/bin/env python3
"""
Local test script for Codesnip tool
This script demonstrates the tool functionality without requiring external APIs
"""

import json
import os
import sys
from unittest.mock import patch, MagicMock

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codesnip.quality_checker import run_all_checks
from codesnip.cli import analyze_code_diff_by_file

def test_quality_checker():
    """Test the quality checker functionality"""
    print("🧪 Testing Quality Checker...")
    checks = run_all_checks()
    
    for tool, result in checks.items():
        print(f"\n📊 {tool.upper()} Results:")
        print(f"Output length: {len(result)} characters")
        if "ERROR" in result.upper() or "FAILED" in result.upper():
            print("❌ Found errors/failures")
        else:
            print("✅ Completed successfully")

def test_code_diff_analyzer():
    """Test the code diff analysis functionality"""
    print("\n🔍 Testing Code Diff Analyzer...")
    
    # Sample diff content for testing
    sample_diff = """diff --git a/example.py b/example.py
index 1234567..abcdefg 100644
--- a/example.py
+++ b/example.py
@@ -1,5 +1,8 @@
 def hello():
+    print("This is a very long line that exceeds the 120 character limit and should be flagged by our code analysis tool")
+    result = eval("2 + 2")  # This should be flagged as unsafe
     return "Hello World"
 
 if __name__ == "__main__":
+    print("Debug output")
     hello()
"""
    
    issues = analyze_code_diff_by_file(sample_diff)
    
    print(f"Found issues in {len(issues)} files:")
    for file, file_issues in issues.items():
        print(f"\n📂 {file}:")
        for issue in file_issues:
            print(f"  ⚠️  {issue}")
    
    return len(issues) > 0

def test_mock_pr_analysis():
    """Test PR analysis with mocked external APIs"""
    print("\n🚀 Testing Mock PR Analysis...")
    
    # Mock PR data
    mock_pr_data = {
        "number": 123,
        "title": "Test PR: Add new feature",
        "body": "This PR adds a new feature for testing purposes.",
        "merged_at": "2024-01-15T10:00:00Z",
        "diff": """diff --git a/src/main.py b/src/main.py
index 1234567..abcdefg 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,5 @@
 def main():
+    print("Starting application")
     print("Hello World")
+    return True
"""
    }
    
    # Mock system metrics
    mock_metrics = {
        "cpu_usage_percent": 25.5,
        "memory_before": 65.2,
        "memory_after": 66.1,
    }
    
    # Mock quality reports
    mock_checks = {
        "pytest": "===== 5 passed in 2.3s =====",
        "coverage": "Coverage: 85%",
        "pylint": "Your code has been rated at 8.5/10",
        "bandit": "No issues identified"
    }
    
    print("📋 Mock PR Data:")
    print(f"  PR #{mock_pr_data['number']}: {mock_pr_data['title']}")
    
    print("\n📈 Mock System Metrics:")
    for key, value in mock_metrics.items():
        print(f"  {key}: {value}")
    
    print("\n🛠️ Mock Quality Checks:")
    for tool, result in mock_checks.items():
        print(f"  {tool}: {result[:50]}...")
    
    # Test code diff analysis
    code_issues = analyze_code_diff_by_file(mock_pr_data["diff"])
    print(f"\n🔍 Code Issues Found: {len(code_issues)} files with issues")
    
    return True

def test_installation():
    """Test that all required dependencies are installed"""
    print("\n📦 Testing Installation Dependencies...")
    
    required_modules = [
        'click', 'requests', 'openai', 'pytest', 
        'coverage', 'pylint', 'bandit', 'psutil'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing modules: {', '.join(missing_modules)}")
        return False
    else:
        print("\n✅ All dependencies installed successfully!")
        return True

def main():
    """Run all local tests"""
    print("🎯 Codesnip Local Testing Suite")
    print("=" * 50)
    
    test_results = []
    
    # Test 1: Installation
    test_results.append(("Installation", test_installation()))
    
    # Test 2: Quality Checker
    try:
        test_quality_checker()
        test_results.append(("Quality Checker", True))
    except Exception as e:
        print(f"❌ Quality Checker test failed: {e}")
        test_results.append(("Quality Checker", False))
    
    # Test 3: Code Diff Analyzer
    try:
        has_issues = test_code_diff_analyzer()
        test_results.append(("Code Diff Analyzer", has_issues))
    except Exception as e:
        print(f"❌ Code Diff Analyzer test failed: {e}")
        test_results.append(("Code Diff Analyzer", False))
    
    # Test 4: Mock PR Analysis
    try:
        success = test_mock_pr_analysis()
        test_results.append(("Mock PR Analysis", success))
    except Exception as e:
        print(f"❌ Mock PR Analysis test failed: {e}")
        test_results.append(("Mock PR Analysis", False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(test_results)} tests passed")
    
    if passed == len(test_results):
        print("🎉 All tests passed! Codesnip is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())