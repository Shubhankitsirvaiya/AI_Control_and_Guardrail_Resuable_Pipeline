import re
from .models import GatewayDecision, RiskLevel, ValidationFinding, ValidationResult, UserRole
from .policy import SafetyPolicy
from .rag import relevance_score


def _contains_any(text: str, patterns: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [pattern for pattern in patterns if pattern in lowered]


def _combine_findings(findings: list[ValidationFinding]) -> ValidationResult:
    if not findings:
        return ValidationResult()

    risk_order = {RiskLevel.low: 1, RiskLevel.medium: 2, RiskLevel.high: 3}
    highest = max((finding.risk_level for finding in findings), key=lambda item: risk_order[item])

    if any(finding.risk_level == RiskLevel.high for finding in findings):
        decision = GatewayDecision.block
    elif any(finding.risk_level == RiskLevel.medium for finding in findings):
        decision = GatewayDecision.escalate
    else:
        decision = GatewayDecision.allow

    return ValidationResult(decision=decision, risk_level=highest, findings=findings)


class DocumentValidator:
    def __init__(self, policy: SafetyPolicy):
        self.policy = policy

    def validate(self, filename: str, content: str) -> ValidationResult:
        findings: list[ValidationFinding] = []
        extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        if extension not in {"txt", "md", "csv", "pdf", "docx"}:
            findings.append(
                ValidationFinding(
                    rule_id="document.unsupported_type",
                    message="Only txt, md, csv, pdf, and docx files are supported in this demo.",
                    risk_level=RiskLevel.medium,
                )
            )

        if len(content) > self.policy.max_document_chars:
            findings.append(
                ValidationFinding(
                    rule_id="document.too_large",
                    message="Document is larger than the configured safety limit.",
                    risk_level=RiskLevel.medium,
                )
            )

        injection_hits = _contains_any(content, self.policy.prompt_injection_patterns)
        if injection_hits:
            findings.append(
                ValidationFinding(
                    rule_id="document.prompt_injection",
                    message=(
                        "Document contains instruction-like text, but uploaded documents are treated as "
                        f"untrusted context: {', '.join(injection_hits[:3])}."
                    ),
                    risk_level=RiskLevel.low,
                )
            )

        sensitive_hits = _contains_any(content, self.policy.sensitive_patterns)
        if sensitive_hits:
            findings.append(
                ValidationFinding(
                    rule_id="document.sensitive_content",
                    message=(
                        "Document may contain sensitive terms. This is allowed for this use case and will "
                        f"be handled by output controls: {', '.join(sensitive_hits[:3])}."
                    ),
                    risk_level=RiskLevel.low,
                )
            )

        return _combine_findings(findings)


class InputValidator:
    def __init__(self, policy: SafetyPolicy):
        self.policy = policy

    def validate(
        self,
        query: str,
        document_text: str | None,
        user_role: UserRole,
        use_case_context: str | None = None,
    ) -> ValidationResult:
        findings: list[ValidationFinding] = []
        normalized = query.strip()

        if len(normalized) < 4:
            findings.append(
                ValidationFinding(
                    rule_id="input.too_short",
                    message="Query is too short to process safely.",
                    risk_level=RiskLevel.low,
                )
            )

        injection_hits = _contains_any(normalized, self.policy.prompt_injection_patterns)
        if injection_hits:
            findings.append(
                ValidationFinding(
                    rule_id="input.prompt_injection",
                    message=f"Prompt injection pattern detected: {', '.join(injection_hits[:3])}.",
                    risk_level=RiskLevel.high,
                )
            )

        sensitive_hits = _contains_any(normalized, self.policy.sensitive_patterns)
        if sensitive_hits:
            findings.append(
                ValidationFinding(
                    rule_id="input.sensitive_credential_request",
                    message=(
                        "Query asks about sensitive credential or personal-data terms and is not allowed: "
                        f"{', '.join(sensitive_hits[:3])}."
                    ),
                    risk_level=RiskLevel.high,
                )
            )

        unsafe_hits = _contains_any(normalized, self.policy.unsafe_advice_patterns)
        if unsafe_hits:
            findings.append(
                ValidationFinding(
                    rule_id="input.regulated_advice",
                    message="Query asks for regulated advice and needs review.",
                    risk_level=RiskLevel.medium,
                )
            )

        if document_text:
            document_score = relevance_score(normalized, document_text)
            use_case_score = relevance_score(normalized, use_case_context or "")
            if (
                document_score < self.policy.relevance_min_score
                and use_case_score < self.policy.use_case_relevance_min_score
            ):
                findings.append(
                    ValidationFinding(
                        rule_id="input.low_relevance",
                        message="Query is not relevant to the selected document or configured use case.",
                        risk_level=RiskLevel.high,
                    )
                )

        result = _combine_findings(findings)

        if result.decision == GatewayDecision.escalate and user_role.value not in self.policy.require_approval_for_roles:
            return ValidationResult(
                decision=GatewayDecision.allow,
                risk_level=result.risk_level,
                findings=result.findings,
            )

        return result


class OutputValidator:
    def __init__(self, policy: SafetyPolicy):
        self.policy = policy

    def validate(self, answer: str, context: list[str]) -> ValidationResult:
        findings: list[ValidationFinding] = []
        sensitive_hits = _contains_any(answer, self.policy.sensitive_patterns)

        if sensitive_hits:
            findings.append(
                ValidationFinding(
                    rule_id="output.sensitive_content",
                    message=(
                        "Answer contains sensitive terms. This is allowed when grounded in public/legal "
                        f"document context: {', '.join(sensitive_hits[:3])}."
                    ),
                    risk_level=RiskLevel.low,
                )
            )

        if re.search(r"\b(always|guaranteed|definitely|100%)\b", answer, flags=re.IGNORECASE):
            findings.append(
                ValidationFinding(
                    rule_id="output.overconfident_claim",
                    message="Answer contains overconfident language and may need review.",
                    risk_level=RiskLevel.low,
                )
            )

        if context and relevance_score(answer, "\n".join(context)) < 0.04:
            findings.append(
                ValidationFinding(
                    rule_id="output.low_grounding",
                    message="Answer appears weakly grounded in retrieved context.",
                    risk_level=RiskLevel.medium,
                )
            )

        return _combine_findings(findings)

