# Safe Response and Approval Gateway

A reusable AI safety control layer for document Q&A applications. The project uses a FastAPI backend, a Streamlit frontend, OpenAI for generation, and a reusable `safe_gateway` module for validation, approvals, audit trails, and feedback.

## Project Idea

This application represents a company-wide guardrail gateway for legal document AI apps. Users can upload legal/company documents, ask questions over approved documents, and receive grounded answers only after the request passes safety checks.

The same gateway can be reused for other business use cases by changing the SME-owned use-case context file instead of rewriting the whole application.

## Features

- Upload and parse `txt`, `md`, `csv`, `pdf`, and `docx` files.
- Preload backend documents from `sample_data/`.
- Validate uploaded documents before Q&A.
- Score document relevance to the configured use case, starting at `50%`.
- Check up to the first `7` parsed pages.
- Allow documents when confidence reaches the configured threshold.
- Use LLM-assisted structured validation for document intake, user queries, and draft outputs.
- Detect prompt injection and unsafe user inputs.
- Run document-grounded chatbot Q&A.
- Use OpenAI embeddings as the primary retrieval path, with keyword retrieval fallback.
- Retrieve relevant chunks only from documents the current user can access.
- Restrict Analysts to their own uploaded documents.
- Allow Manager, Risk Reviewer, and Admin to query all documents only when they enable the sidebar toggle.
- Validate model outputs before display.
- Route risky requests to approval queue.
- Support role switching: `Analyst`, `Manager`, `Risk Reviewer`, `Admin`.
- Store audit trail events in SQLite.
- Store hidden prompt audit records.
- Collect positive/negative answer feedback.
- Use reusable prompts plus one SME-maintained context file.
- Use LangChain output parsers for final answer text and structured validation output.
- Use LLM validation as the primary nuanced safety check; deterministic rules remain as hard-block backup and fallback.

## Architecture

```text
safe_gateway/        reusable safety, validation, RAG, audit, approval logic
api/                 FastAPI backend
streamlit_app.py     Streamlit frontend that consumes the FastAPI API
prompts/             reusable prompt library and SME use-case context
sample_data/         backend documents loaded at startup
data/                local SQLite database, created at runtime
```

## Setup

Create and activate a virtual environment if you want one:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your local environment file:

```bash
copy .env.example .env
```

Update `.env`:

```text
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GATEWAY_DB_PATH=data/gateway.db
DOCUMENT_USE_CASE_CONTEXT_PATH=prompts/use_case_legal_context.txt
DOCUMENT_CONFIDENCE_THRESHOLD=45
```

## Run The Application

Start the FastAPI backend:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Start the Streamlit frontend in another terminal:

```bash
python -m streamlit run streamlit_app.py
```

Then open the Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

FastAPI runs at:

```text
http://127.0.0.1:8000
```

FastAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

## Troubleshooting

If FastAPI fails with this error:

```text
ModuleNotFoundError: No module named 'exceptions'
```

you likely have the old `docx` package installed. Remove it and install `python-docx`:

```bash
python -m pip uninstall docx -y
python -m pip install python-docx
```

Then start the API again:

```bash
python -m uvicorn api.main:app --reload --port 8000
```

## Demo Flow

1. Start FastAPI.
2. Start Streamlit.
3. Select a role from the sidebar.
4. Upload a document.
5. Ask a question in the chatbot.
6. The gateway retrieves relevant chunks only from that user's accessible documents.
7. Manager, Risk Reviewer, and Admin can turn on `Query all documents` when they need cross-document access.
8. If the request is escalated, switch to `Risk Reviewer`, `Manager`, or `Admin`.
9. Approve or reject the request.
10. Switch to `Admin` to inspect audit trail, hidden prompt audits, and feedback.
11. Mark answers as `Good answer` or `Needs improvement`.

## Reusable Prompt Pattern

The core prompt templates live in `prompts/`:

- `answer_system.txt`
- `answer_user.txt`
- `document_validation_system.txt`
- `document_validation_user.txt`

The business-specific rules live in one SME-owned context file:

- `use_case_legal_context.txt`

To reuse the gateway for another use case, copy:

```text
prompts/use_case_context_template.txt
```

Fill it with the SME's business scope, allowed documents, restricted behavior, and relevance signals. Then update `.env`:

```text
DOCUMENT_USE_CASE_CONTEXT_PATH=prompts/your_new_context_file.txt
```

No model training is needed. The guardrail layer stays the same; only the use-case context changes.

## Guardrails Included

- Document type validation.
- Document prompt-injection detection.
- Document use-case confidence scoring.
- Document sensitive content indicators are logged without automatically blocking public/legal documents.
- Input prompt-injection detection.
- User queries asking for credentials, secrets, passwords, tokens, or personal data are blocked.
- LLM-assisted input validation for nuanced relevance, intent, and escalation decisions.
- Query relevance checks.
- Query relevance checks use both the selected document and the SME use-case context.
- User-level document access checks.
- Privileged all-document access requires an explicit UI toggle.
- Hidden system prompts.
- Output validation.
- LLM-assisted output validation for grounding, prompt leakage, and unsafe final answers.
- Approval routing.
- Audit logging.
- Prompt audit logging.
- User feedback collection.

