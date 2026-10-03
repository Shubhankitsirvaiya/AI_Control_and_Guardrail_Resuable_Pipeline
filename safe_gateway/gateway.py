from uuid import uuid4
from pathlib import Path

from .approval import ApprovalQueue
from .audit import AuditStore
from .document_confidence import DocumentConfidenceEvaluator
from .document_parser import parse_document_with_pages
from .llm import OpenAIAnswerGenerator
from .llm_safety import LLMSafetyEvaluator
from .models import (
    ApprovalItem,
    ChatRequest,
    ChatResponse,
    DocumentRecord,
    FeedbackRequest,
    GatewayDecision,
    RiskLevel,
    UserRole,
    ValidationFinding,
)
from .policy import SafetyPolicy
from .rag import OpenAIEmbeddingRetriever, retrieve
from .validators import DocumentValidator, InputValidator, OutputValidator


class SafetyGateway:
    def __init__(self, audit_store: AuditStore | None = None, policy: SafetyPolicy | None = None):
        self.policy = policy or SafetyPolicy()
        self.audit = audit_store or AuditStore()
        self.answer_generator = OpenAIAnswerGenerator()
        self.embedding_retriever = OpenAIEmbeddingRetriever()
        self.document_confidence = DocumentConfidenceEvaluator()
        self.llm_safety = LLMSafetyEvaluator()
        self.approvals = ApprovalQueue()
        self.documents: dict[str, DocumentRecord] = {}
        self.document_validator = DocumentValidator(self.policy)
        self.input_validator = InputValidator(self.policy)
        self.output_validator = OutputValidator(self.policy)
        self._load_stored_documents()
        self.load_backend_documents()

    def _merge_validation_results(self, *results: ValidationResult) -> ValidationResult:
        findings = [finding for result in results for finding in result.findings]
        risk_rank = {RiskLevel.low: 1, RiskLevel.medium: 2, RiskLevel.high: 3}
        decision_rank = {
            GatewayDecision.allow: 1,
            GatewayDecision.escalate: 2,
            GatewayDecision.block: 3,
        }
        highest_risk = max((result.risk_level for result in results), key=lambda risk: risk_rank[risk])
        strictest_decision = max((result.decision for result in results), key=lambda decision: decision_rank[decision])
        if not findings and strictest_decision == GatewayDecision.allow:
            return ValidationResult()
        return ValidationResult(
            decision=strictest_decision,
            risk_level=highest_risk,
            findings=findings,
        )

    def _load_stored_documents(self) -> None:
        for document in self.audit.list_documents():
            self.documents[document.id] = document

    def load_backend_documents(self, directory: str = "sample_data") -> None:
        sample_dir = Path(directory)
        if not sample_dir.exists():
            return

        existing_backend_names = {
            document.name for document in self.documents.values() if document.source == "backend"
        }
        for path in sample_dir.iterdir():
            if not path.is_file() or path.name in existing_backend_names:
                continue
            try:
                parsed = parse_document_with_pages(path.name, path.read_bytes())
            except ValueError:
                continue
            self.add_document(
                path.name,
                parsed.content,
                source="backend",
                document_id=f"backend-{path.stem}",
                pages=parsed.pages,
                owner_id="system",
            )

    def add_document(
        self,
        filename: str,
        content: str,
        source: str = "uploaded",
        document_id: str | None = None,
        pages: list[str] | None = None,
        owner_id: str = "system",
    ) -> DocumentRecord:
        document_id = document_id or str(uuid4())
        validation = self.document_validator.validate(filename, content)

        confidence = self.document_confidence.evaluate(filename, pages or [content])
        self.audit.log_prompt(
            request_id=document_id,
            model=confidence.prompt_record["model"],
            system_prompt=confidence.prompt_record["system_prompt"],
            user_prompt=confidence.prompt_record["user_prompt"],
        )
        validation.confidence_score = confidence.confidence
        validation.confidence_summary = confidence.summary
        validation.findings.append(
            ValidationFinding(
                rule_id="document.use_case_confidence",
                message=(
                    f"Document use-case confidence is {confidence.confidence}%. "
                    f"Threshold is {self.document_confidence.confidence_threshold}%. {confidence.summary}"
                ),
                risk_level=RiskLevel.low if confidence.allowed else RiskLevel.high,
            )
        )
        if not confidence.allowed:
            validation.decision = GatewayDecision.block
            validation.risk_level = RiskLevel.high

        record = DocumentRecord(
            id=document_id,
            name=filename,
            content=content,
            validation=validation,
            source=source,
            owner_id=owner_id,
        )
        self.documents[document_id] = record
        self.audit.save_document(record)
        self.audit.log(
            action="DOCUMENT_REGISTERED" if source == "backend" else "DOCUMENT_UPLOADED",
            request_id=document_id,
            decision=validation.decision.value,
            risk_level=validation.risk_level.value,
            detail=f"{filename} ({source}, owner={owner_id})",
        )
        return record

    def list_documents(self) -> list[DocumentRecord]:
        return list(self.documents.values())

    def list_documents_for_user(
        self,
        user_id: str,
        user_role: UserRole,
        allow_all_documents: bool = False,
    ) -> list[DocumentRecord]:
        if allow_all_documents and user_role in {UserRole.manager, UserRole.risk_reviewer, UserRole.admin}:
            return list(self.documents.values())
        return [document for document in self.documents.values() if document.owner_id == user_id]

    def revalidate_documents(self) -> list[DocumentRecord]:
        current_documents = list(self.documents.values())
        self.documents = {}
        revalidated: list[DocumentRecord] = []
        for document in current_documents:
            revalidated.append(
                self.add_document(
                    document.name,
                    document.content,
                    source=document.source,
                    document_id=document.id,
                    pages=[document.content],
                    owner_id=document.owner_id,
                )
            )
        return revalidated

    def chat(self, request: ChatRequest) -> ChatResponse:
        request_id = str(uuid4())
        accessible_documents = self.list_documents_for_user(
            request.user_id,
            request.user_role,
            request.allow_all_documents,
        )
        if request.document_id:
            accessible_documents = [
                document for document in accessible_documents if document.id == request.document_id
            ]

        self.audit.log(
            action="QUERY_RECEIVED",
            request_id=request_id,
            user_role=request.user_role.value,
            detail=f"{request.query} (user={request.user_id}, all_docs={request.allow_all_documents})",
        )

        if not accessible_documents:
            self.audit.log(
                action="QUERY_BLOCKED",
                request_id=request_id,
                user_role=request.user_role.value,
                decision=GatewayDecision.block.value,
                risk_level=RiskLevel.medium.value,
                detail="No accessible documents.",
            )
            return ChatResponse(
                request_id=request_id,
                decision=GatewayDecision.block,
                risk_level=RiskLevel.medium,
                reasons=["Upload a document first, or enable all-document access if your role allows it."],
            )

        blocked_documents = [
            document for document in accessible_documents if document.validation.decision == GatewayDecision.block
        ]
        searchable_documents = [
            document for document in accessible_documents if document.validation.decision != GatewayDecision.block
        ]

        if not searchable_documents:
            reasons = [
                finding.message
                for document in blocked_documents
                for finding in document.validation.findings
            ]
            self.audit.log(
                action="QUERY_BLOCKED_UNSAFE_DOCUMENT",
                request_id=request_id,
                user_role=request.user_role.value,
                decision=GatewayDecision.block.value,
                risk_level=RiskLevel.high.value,
                detail="; ".join(reasons),
            )
            return ChatResponse(
                request_id=request_id,
                decision=GatewayDecision.block,
                risk_level=RiskLevel.high,
                reasons=["The selected document failed safety validation."] + reasons,
            )

        document_text = "\n\n".join(
            f"Document: {document.name}\n{document.content}" for document in searchable_documents
        )

        input_result = self.input_validator.validate(
            request.query,
            document_text,
            request.user_role,
            use_case_context=self.document_confidence.prompts.load_use_case_context(),
        )
        llm_input_result = self.llm_safety.validate_input(
            query=request.query,
            user_role=request.user_role,
            document_text=document_text,
        )
        self.audit.log_prompt(
            request_id=request_id,
            model=llm_input_result.prompt_record["model"],
            system_prompt=llm_input_result.prompt_record["system_prompt"],
            user_prompt=llm_input_result.prompt_record["user_prompt"],
        )
        input_result = self._merge_validation_results(input_result, llm_input_result.validation)
        reasons = [finding.message for finding in input_result.findings]

        if input_result.decision == GatewayDecision.block:
            self.audit.log(
                action="INPUT_BLOCKED",
                request_id=request_id,
                user_role=request.user_role.value,
                decision=input_result.decision.value,
                risk_level=input_result.risk_level.value,
                detail="; ".join(reasons),
            )
            return ChatResponse(
                request_id=request_id,
                decision=GatewayDecision.block,
                risk_level=input_result.risk_level,
                reasons=reasons,
            )

        contexts: list[str] = []
        for document in searchable_documents:
            for chunk in retrieve(request.query, document.content, top_k=2, embedding_retriever=self.embedding_retriever):
                contexts.append(f"[{document.name}] {chunk}")
        contexts = contexts[:5]
        draft_answer, prompt_record = self.answer_generator.generate(request.query, contexts)
        self.audit.log_prompt(
            request_id=request_id,
            model=prompt_record["model"],
            system_prompt=prompt_record["system_prompt"],
            user_prompt=prompt_record["user_prompt"],
        )
        output_result = self.output_validator.validate(draft_answer, contexts)
        llm_output_result = self.llm_safety.validate_output(
            query=request.query,
            answer=draft_answer,
            retrieved_context=contexts,
        )
        self.audit.log_prompt(
            request_id=request_id,
            model=llm_output_result.prompt_record["model"],
            system_prompt=llm_output_result.prompt_record["system_prompt"],
            user_prompt=llm_output_result.prompt_record["user_prompt"],
        )
        output_result = self._merge_validation_results(output_result, llm_output_result.validation)
        reasons.extend(finding.message for finding in output_result.findings)

        decision = input_result.decision
        risk_level = input_result.risk_level
        if output_result.decision == GatewayDecision.block:
            decision = GatewayDecision.block
            risk_level = output_result.risk_level
        elif output_result.decision == GatewayDecision.escalate:
            decision = GatewayDecision.escalate
            risk_level = output_result.risk_level

        if decision == GatewayDecision.block:
            self.audit.log(
                action="OUTPUT_BLOCKED",
                request_id=request_id,
                user_role=request.user_role.value,
                decision=decision.value,
                risk_level=risk_level.value,
                detail="; ".join(reasons),
            )
            return ChatResponse(
                request_id=request_id,
                decision=decision,
                risk_level=risk_level,
                reasons=reasons,
                retrieved_context=contexts,
            )

        if decision == GatewayDecision.escalate:
            approval_id = str(uuid4())
            self.approvals.add(
                ApprovalItem(
                    id=approval_id,
                    request_id=request_id,
                    document_id=request.document_id,
                    requester_user_id=request.user_id,
                    user_role=request.user_role,
                    query=request.query,
                    draft_answer=draft_answer,
                    reasons=reasons,
                    risk_level=risk_level,
                    retrieved_context=contexts,
                    allow_all_documents=request.allow_all_documents,
                )
            )
            self.audit.log(
                action="APPROVAL_REQUESTED",
                request_id=request_id,
                user_role=request.user_role.value,
                decision=decision.value,
                risk_level=risk_level.value,
                detail=f"approval_id={approval_id}; requester={request.user_id}; " + "; ".join(reasons),
            )
            return ChatResponse(
                request_id=request_id,
                decision=decision,
                risk_level=risk_level,
                reasons=reasons,
                approval_id=approval_id,
                retrieved_context=contexts,
            )

        self.audit.log(
            action="RESPONSE_DELIVERED",
            request_id=request_id,
            user_role=request.user_role.value,
            decision=GatewayDecision.allow.value,
            risk_level=risk_level.value,
            detail="; ".join(reasons) if reasons else "Passed safety checks.",
        )
        return ChatResponse(
            request_id=request_id,
            decision=GatewayDecision.allow,
            risk_level=risk_level,
            answer=draft_answer,
            reasons=reasons or ["Passed safety checks."],
            retrieved_context=contexts,
        )

    def decide_approval(self, approval_id: str, approved: bool, reviewer_role: UserRole) -> ChatResponse:
        if reviewer_role not in {UserRole.manager, UserRole.risk_reviewer, UserRole.admin}:
            raise PermissionError("Only Manager, Risk Reviewer, or Admin can decide approvals.")

        item = self.approvals.decide(approval_id, approved, reviewer_role)
        action = "QUERY_APPROVED" if approved else "QUERY_REJECTED"
        decision = GatewayDecision.allow if approved else GatewayDecision.block
        self.audit.log(
            action=action,
            request_id=item.request_id,
            user_role=reviewer_role.value,
            decision=decision.value,
            risk_level=item.risk_level.value,
            detail=(
                f"Reviewer {reviewer_role.value} decided approval {approval_id}. "
                f"Requester={item.requester_user_id}; request_role={item.user_role.value}."
            ),
        )

        if not approved:
            return ChatResponse(
                request_id=item.request_id,
                decision=GatewayDecision.block,
                risk_level=item.risk_level,
                reasons=["The reviewer rejected this request."],
                approval_id=approval_id,
            )

        self.audit.log(
            action="RESPONSE_DELIVERED_AFTER_APPROVAL",
            request_id=item.request_id,
            user_role=item.user_role.value,
            decision=GatewayDecision.allow.value,
            risk_level=item.risk_level.value,
            detail="Approved response released.",
        )
        return ChatResponse(
            request_id=item.request_id,
            decision=GatewayDecision.allow,
            risk_level=item.risk_level,
            answer=item.draft_answer,
            reasons=["Approved by reviewer."],
            approval_id=approval_id,
            retrieved_context=item.retrieved_context,
        )

    def add_feedback(self, feedback: FeedbackRequest) -> None:
        normalized = feedback.rating.lower().strip()
        rating = "positive" if normalized in {"yes", "positive", "good", "up"} else "negative"
        self.audit.add_feedback(feedback.request_id, rating, feedback.comment)

