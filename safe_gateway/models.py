from enum import Enum
from pydantic import BaseModel, Field


class UserRole(str, Enum):
    analyst = "Analyst"
    manager = "Manager"
    risk_reviewer = "Risk Reviewer"
    admin = "Admin"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class GatewayDecision(str, Enum):
    allow = "ALLOW"
    block = "BLOCK"
    escalate = "ESCALATE_FOR_APPROVAL"
    redact = "REDACT"
    regenerate = "REGENERATE"


class ApprovalDecision(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class ValidationFinding(BaseModel):
    rule_id: str
    message: str
    risk_level: RiskLevel = RiskLevel.low


class ValidationResult(BaseModel):
    decision: GatewayDecision = GatewayDecision.allow
    risk_level: RiskLevel = RiskLevel.low
    findings: list[ValidationFinding] = Field(default_factory=list)
    confidence_score: int | None = None
    confidence_summary: str | None = None


class DocumentRecord(BaseModel):
    id: str
    name: str
    content: str
    validation: ValidationResult
    source: str = "uploaded"
    owner_id: str = "system"


class ChatRequest(BaseModel):
    user_role: UserRole
    query: str
    user_id: str
    document_id: str | None = None
    allow_all_documents: bool = False


class ChatResponse(BaseModel):
    request_id: str
    decision: GatewayDecision
    risk_level: RiskLevel
    answer: str | None = None
    reasons: list[str] = Field(default_factory=list)
    approval_id: str | None = None
    retrieved_context: list[str] = Field(default_factory=list)


class ApprovalItem(BaseModel):
    id: str
    request_id: str
    document_id: str | None
    requester_user_id: str
    user_role: UserRole
    query: str
    draft_answer: str | None
    reasons: list[str]
    risk_level: RiskLevel
    retrieved_context: list[str] = Field(default_factory=list)
    allow_all_documents: bool = False
    status: ApprovalDecision = ApprovalDecision.pending
    reviewer_role: UserRole | None = None


class FeedbackRequest(BaseModel):
    request_id: str
    rating: str
    comment: str | None = None

