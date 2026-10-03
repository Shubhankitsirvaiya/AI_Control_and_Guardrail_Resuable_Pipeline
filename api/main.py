from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from safe_gateway import ChatRequest, FeedbackRequest, SafetyGateway, UserRole
from safe_gateway.document_parser import parse_document_with_pages


app = FastAPI(title="Safe Response and Approval Gateway")
gateway = SafetyGateway()


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    reviewer_role: UserRole


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...),
):
    raw = await file.read()
    try:
        parsed = parse_document_with_pages(file.filename or "uploaded.txt", raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return gateway.add_document(
        file.filename or "uploaded.txt",
        parsed.content,
        pages=parsed.pages,
        owner_id=user_id,
    )


@app.get("/documents")
def list_documents(
    user_id: str | None = None,
    user_role: UserRole = UserRole.analyst,
    allow_all_documents: bool = False,
):
    if user_id:
        return gateway.list_documents_for_user(user_id, user_role, allow_all_documents)
    return gateway.list_documents()


@app.post("/documents/reload-backend")
def reload_backend_documents():
    before = len(gateway.list_documents())
    gateway.load_backend_documents()
    after = len(gateway.list_documents())
    return {"status": "reloaded", "added": after - before, "total": after}


@app.post("/documents/revalidate")
def revalidate_documents():
    documents = gateway.revalidate_documents()
    return {"status": "revalidated", "total": len(documents)}


@app.post("/chat")
def chat(request: ChatRequest):
    return gateway.chat(request)


@app.get("/approvals")
def list_approvals():
    return gateway.approvals.list_all()


@app.post("/approvals/{approval_id}/decision")
def decide_approval(approval_id: str, request: ApprovalDecisionRequest):
    try:
        return gateway.decide_approval(approval_id, request.approved, request.reviewer_role)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/audit")
def list_audit():
    return gateway.audit.list_audit_events()


@app.get("/prompt-audit")
def list_prompt_audit():
    return gateway.audit.list_prompt_audits()


@app.post("/feedback")
def add_feedback(request: FeedbackRequest):
    gateway.add_feedback(request)
    return {"status": "recorded"}


@app.get("/feedback")
def list_feedback():
    return gateway.audit.list_feedback()

