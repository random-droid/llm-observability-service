import re
from typing import List, Dict
from app.models import SecurityIssue

# Patterns for detection
SECRET_PATTERNS = {
    'aws_access_key': {
        'pattern': r'AKIA[0-9A-Z]{16}',
        'severity': 'critical'
    },
    'github_token': {
        'pattern': r'gh[ps]_[A-Za-z0-9]{36}',
        'severity': 'critical'
    },
    'private_key': {
        'pattern': r'-----BEGIN.*PRIVATE KEY-----',
        'severity': 'critical'
    },
    'api_key': {
        'pattern': r'api[_-]?key["\s:=]+[A-Za-z0-9]{20,}',
        'severity': 'high'
    },
    'slack_token': {
        'pattern': r'xox[baprs]-[0-9]{10,12}-[0-9]{10,12}-[A-Za-z0-9]{24,}',
        'severity': 'critical'
    }
}

PII_PATTERNS = {
    'email': {
        'pattern': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'severity': 'medium'
    },
    'ssn': {
        'pattern': r'\b\d{3}-\d{2}-\d{4}\b',
        'severity': 'high'
    },
    'phone': {
        'pattern': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'severity': 'low'
    },
    'credit_card': {
        'pattern': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        'severity': 'high'
    }
}

INJECTION_PATTERNS = {
    'ignore_instructions': {
        'pattern': r'ignore\s+(previous|all|above)\s+instructions?',
        'severity': 'high'
    },
    'disregard': {
        'pattern': r'disregard.*(above|previous|earlier)',
        'severity': 'high'
    },
    'new_instructions': {
        'pattern': r'(you are now|new instructions?:)',
        'severity': 'high'
    },
    'system_prompt': {
        'pattern': r'system\s+prompt',
        'severity': 'medium'
    }
}

def detect_secrets(text: str) -> List[SecurityIssue]:
    """Detect secrets in text"""
    issues = []
    for secret_type, config in SECRET_PATTERNS.items():
        if re.search(config['pattern'], text, re.IGNORECASE):
            issues.append(SecurityIssue(
                type=f"secret_{secret_type}",
                severity=config['severity'],
                details=f"Detected {secret_type.replace('_', ' ')}"
            ))
    return issues

def detect_pii(text: str) -> List[SecurityIssue]:
    """Detect PII in text"""
    issues = []
    for pii_type, config in PII_PATTERNS.items():
        if re.search(config['pattern'], text, re.IGNORECASE):
            issues.append(SecurityIssue(
                type=f"pii_{pii_type}",
                severity=config['severity'],
                details=f"Detected {pii_type.replace('_', ' ')}"
            ))
    return issues

def detect_prompt_injection(text: str) -> List[SecurityIssue]:
    """Detect prompt injection attempts"""
    issues = []
    for injection_type, config in INJECTION_PATTERNS.items():
        if re.search(config['pattern'], text, re.IGNORECASE):
            issues.append(SecurityIssue(
                type=f"prompt_injection_{injection_type}",
                severity=config['severity'],
                details=f"Potential prompt injection: {injection_type.replace('_', ' ')}"
            ))
    return issues

def run_all_security_checks(text: str) -> List[SecurityIssue]:
    """Run all security checks"""
    issues = []
    issues.extend(detect_secrets(text))
    issues.extend(detect_pii(text))
    issues.extend(detect_prompt_injection(text))
    return issues