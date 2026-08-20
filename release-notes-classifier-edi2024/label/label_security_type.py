"""
classify_security_type.py

Keyword-based classifier for "Security Risk" (see Berhe et al., "Triage
Software Update Impact via Release Notes Classification," Procedia Computer
Science, 2024).

Pure classification logic, no external dependencies — pass any release-note
text to classify_text() and get back the matching security-related category
labels (plus the text with matched keywords bolded, for inspection). If you
independently know a release is tied to a CVE, force-add "SECURITY" to its
labels the way the production pipeline does (see `is_cve` note below).
"""

import re

# Define Software Legal, Privacy, Safety, and Security Categories
categories_security = {
    "legal": [
        "license", "GPL", "MIT", "Apache License", "BSD", "copyright", "terms of service",
        "EULA", "patent", "intellectual property", "trademark", "act", "DMCA", "compliance",
        "open source", "permissive license", "copyleft", "derivative works", "contract",
        "agreement", "litigation", "liability", "fair use", "copyright infringement"
    ],
    "privacy": [
        "GDPR", "CCPA", "privacy policy", "data protection", "user consent", "cookies",
        "personal data", "anonymization", "right to be forgotten", "PII", "data breach",
        "privacy by design", "HIPAA", "data minimization", "third-party data", "tracking",
        "surveillance", "behavioral analytics", "data retention", "opt-out", "cookie banner"
    ],
    "safety": [
        "compliance", "safety standards", "ISO", "NIST", "OSHA", "incident response",
        "product safety", "physical security", "hazard", "critical infrastructure",
        "safety compliance", "quality assurance", "workplace safety", "incident management",
        "disaster recovery", "business continuity", "risk assessment", "public safety",
        "automotive safety", "aviation safety", "safety-critical systems", "fire safety"
    ],
    "security": [
        "cybersecurity", "credential", "token", "encryption", "vulnerability", "CVE",
        "malware", "ransomware", "penetration testing", "secure coding", "firewall",
        "zero trust", "IAM", "phishing", "data exfiltration", "APT", "supply chain attack",
        "exploit", "buffer overflow", "MITRE ATT&CK", "NIST Cybersecurity Framework",
        "SOC2", "SIEM", "hacking", "black hat", "white hat", "penetration test", "security audit",
        "insider threat", "zero-day", "patch management", "application security",
        "red team", "blue team", "threat intelligence", "data leak", "DDoS", "SOC", "SIEM",
        "XSS", "CSRF", "man-in-the-middle attack", "TLS", "HTTPS", "WAF", "endpoint security", "virus"
    ],
    "compliance": [
        "PCI DSS", "SOX", "HIPAA", "FISMA", "ISO 27001", "ISO 9001", "SOC 1", "SOC 2",
        "GDPR compliance", "CCPA compliance", "audit log", "regulatory compliance",
        "financial compliance", "compliance risk", "compliance management", "internal audit"
    ],
    "fraud": [
        "fraud detection", "fraud prevention", "identity theft", "credit card fraud",
        "phishing scam", "social engineering", "money laundering", "deepfake", "fake identity",
        "credential stuffing", "data fraud", "account takeover", "fraud analytics", "malvertising"
    ],
    "trust_and_safety": [
        "user verification", "content moderation", "misinformation", "disinformation",
        "platform abuse", "hate speech detection", "spam filtering", "bot detection",
        "account security", "user protection", "CSAM detection", "abuse prevention"
    ],
    "identity_and_access": [
        "authentication", "authorization", "MFA", "2FA", "SSO", "OAuth", "SAML", "JWT",
        "RBAC", "ABAC", "biometric authentication", "federated identity", "identity provider",
        "user provisioning", "directory service", "LDAP", "passwordless login"
    ],
    "risk_management": [
        "risk assessment", "threat modeling", "security posture", "business continuity",
        "risk mitigation", "incident response planning", "forensic analysis", "compliance risk",
        "financial risk", "operational risk", "third-party risk", "risk scoring", "threat detection"
    ],
    "forensics": [
        "digital forensics", "evidence collection", "incident investigation", "log analysis",
        "malware analysis", "memory forensics", "disk forensics", "network forensics",
        "cybercrime investigation", "law enforcement", "chain of custody", "threat attribution"
    ],
    "cloud_security": [
        "cloud security", "CSPM", "CASB", "cloud encryption", "SaaS security", "IaaS security",
        "cloud access control", "zero trust architecture", "hybrid cloud", "data sovereignty",
        "multi-cloud security", "serverless security", "API security", "cloud compliance"
    ]
}


def classify_text(text, is_cve=False):
    """
    Return (labels, highlighted_text) for the Security Risk category.

    `is_cve`: pass True if you separately know this release note is tied to a
    CVE — the production pipeline force-adds "SECURITY" (and drops "UNKNOWN")
    in that case.
    """
    if not text:
        return ["UNKNOWN"], text

    words = re.findall(r'\b\w+\b', text.lower())
    matches = set()
    highlighted_text = text

    for word in words:
        for category, keywords in categories_security.items():
            if word.lower() in (kw.lower() for kw in keywords):
                matches.add(category.upper())
                highlighted_text = re.sub(r'\b' + re.escape(word) + r'\b', f"**{word}**", highlighted_text, flags=re.IGNORECASE)

    labels = list(matches) if matches else ["UNKNOWN"]

    if is_cve:
        labels.append("SECURITY")
        labels = list(set(labels))
        labels = [label for label in labels if label.lower() != "unknown"]

    return labels, highlighted_text


if __name__ == "__main__":
    sample = "This release patches a critical vulnerability (CVE-2024-12345) allowing remote code execution."
    labels, highlighted = classify_text(sample)
    print(labels)
    print(highlighted)
