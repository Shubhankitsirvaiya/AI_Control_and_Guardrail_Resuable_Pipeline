import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
ROLES = ["Analyst", "Manager", "Risk Reviewer", "Admin"]
PRIVILEGED_ROLES = {"Manager", "Risk Reviewer", "Admin"}


st.set_page_config(page_title="AI Safety Gateway", layout="wide")


def user_id_for(role: str) -> str:
    return role.lower().replace(" ", "_")


def api_get(path: str, params: dict | None = None):
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def try_api_post(path: str, payload: dict):
    try:
        return api_post(path, payload), None
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        return None, detail


def upload_document(file, user_id: str):
    files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
    data = {"user_id": user_id}
    response = requests.post(f"{API_BASE_URL}/documents", files=files, data=data, timeout=60)
    response.raise_for_status()
    return response.json()


def show_message(role: str, content: str):
    with st.chat_message(role):
        st.write(content)


st.title("Safe Response and Approval Gateway")

with st.sidebar:
    selected_role = st.selectbox("User", ROLES)
    current_user_id = user_id_for(selected_role)
    allow_all_documents = False

    if selected_role in PRIVILEGED_ROLES:
        allow_all_documents = st.toggle("Query all documents", value=False)

    uploaded = st.file_uploader("Upload document", type=["txt", "md", "csv", "pdf", "docx"])
    if uploaded:
        upload_signature = f"{current_user_id}:{uploaded.name}:{uploaded.size}"
        if st.session_state.get("last_upload_signature") != upload_signature:
            try:
                document = upload_document(uploaded, current_user_id)
                st.session_state["last_upload_signature"] = upload_signature
                st.success(f"Uploaded {document['name']}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    try:
        documents = api_get(
            "/documents",
            params={
                "user_id": current_user_id,
                "user_role": selected_role,
                "allow_all_documents": allow_all_documents,
            },
        )
    except Exception:
        documents = []

    st.caption(f"{len(documents)} accessible document(s)")

tab_names = ["Chat"]
if selected_role in PRIVILEGED_ROLES:
    tab_names.append("Approvals")
if selected_role == "Admin":
    tab_names.extend(["Audit Trail", "Feedback"])

tabs = st.tabs(tab_names)

with tabs[0]:
    messages_key = f"messages_{current_user_id}"
    if messages_key not in st.session_state:
        st.session_state[messages_key] = []

    for message in st.session_state[messages_key]:
        show_message(message["role"], message["content"])

    prompt = st.chat_input("Ask a question about your uploaded documents")
    if prompt:
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        show_message("user", prompt)

        with st.chat_message("assistant"):
            with st.spinner("Checking guardrails and searching relevant chunks..."):
                try:
                    result = api_post(
                        "/chat",
                        {
                            "user_role": selected_role,
                            "user_id": current_user_id,
                            "query": prompt,
                            "allow_all_documents": allow_all_documents,
                        },
                    )
                except Exception as exc:
                    result = {
                        "decision": "BLOCK",
                        "risk_level": "high",
                        "answer": None,
                        "reasons": [str(exc)],
                        "retrieved_context": [],
                    }

            if result["decision"] == "ALLOW" and result.get("answer"):
                content = result["answer"]
            elif result["decision"] == "ESCALATE_FOR_APPROVAL":
                content = "This request needs approval before the answer can be released."
            else:
                content = "I cannot answer that request.\n\n" + "\n".join(result.get("reasons", []))

            st.write(content)
            st.session_state["last_response"] = result
            st.session_state[messages_key].append({"role": "assistant", "content": content})

            if result.get("retrieved_context"):
                with st.expander("Relevant chunks"):
                    for idx, chunk in enumerate(result["retrieved_context"], start=1):
                        st.markdown(f"**Chunk {idx}**")
                        st.write(chunk)

            if result.get("answer"):
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Good answer", use_container_width=True):
                        api_post("/feedback", {"request_id": result["request_id"], "rating": "positive"})
                        st.success("Feedback recorded.")
                with col_b:
                    if st.button("Needs improvement", use_container_width=True):
                        api_post("/feedback", {"request_id": result["request_id"], "rating": "negative"})
                        st.warning("Feedback recorded.")

if selected_role in PRIVILEGED_ROLES:
    approval_tab_index = tab_names.index("Approvals")
    with tabs[approval_tab_index]:
        st.subheader("Approval Queue")
        try:
            approvals = api_get("/approvals")
        except Exception:
            approvals = []

        if not approvals:
            st.info("No approval requests yet.")

        for item in approvals:
            with st.container(border=True):
                st.write(f"**Status:** {item['status']} | **Risk:** {item['risk_level']}")
                st.write(f"**Requested by:** {item.get('requester_user_id', 'unknown')} ({item['user_role']})")
                st.write(f"**Scope:** {'All documents' if item.get('allow_all_documents') else 'Own documents'}")
                st.write(f"**Query:** {item['query']}")
                for reason in item["reasons"]:
                    st.caption(reason)
                if item.get("draft_answer"):
                    with st.expander("Draft answer"):
                        st.write(item["draft_answer"])
                if item.get("retrieved_context"):
                    with st.expander("Retrieved context"):
                        for idx, chunk in enumerate(item["retrieved_context"], start=1):
                            st.markdown(f"**Chunk {idx}**")
                            st.write(chunk)

                if item["status"] == "pending":
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Approve", key=f"approve-{item['id']}", use_container_width=True):
                            response, error = try_api_post(
                                f"/approvals/{item['id']}/decision",
                                {"approved": True, "reviewer_role": selected_role},
                            )
                            if error:
                                st.error(error)
                            else:
                                st.session_state["last_response"] = response
                                st.rerun()
                    with col_b:
                        if st.button("Reject", key=f"reject-{item['id']}", use_container_width=True):
                            response, error = try_api_post(
                                f"/approvals/{item['id']}/decision",
                                {"approved": False, "reviewer_role": selected_role},
                            )
                            if error:
                                st.error(error)
                            else:
                                st.session_state["last_response"] = response
                                st.rerun()

if selected_role == "Admin":
    with tabs[tab_names.index("Audit Trail")]:
        st.subheader("Audit Trail")
        try:
            audit_events = api_get("/audit")
        except Exception:
            audit_events = []

        if audit_events:
            st.dataframe(audit_events, use_container_width=True, hide_index=True)
        else:
            st.info("No audit events recorded yet.")

        with st.expander("Hidden prompt audit"):
            try:
                prompt_audits = api_get("/prompt-audit")
            except Exception:
                prompt_audits = []
            if prompt_audits:
                st.dataframe(prompt_audits, use_container_width=True, hide_index=True)
            else:
                st.info("No hidden prompts have been audited yet.")

    with tabs[tab_names.index("Feedback")]:
        st.subheader("Feedback")
        try:
            feedback = api_get("/feedback")
        except Exception:
            feedback = []

        if feedback:
            positive = sum(1 for item in feedback if item["rating"] == "positive")
            negative = sum(1 for item in feedback if item["rating"] == "negative")
            col_1, col_2 = st.columns(2)
            col_1.metric("Positive", positive)
            col_2.metric("Negative", negative)
            st.dataframe(feedback, use_container_width=True, hide_index=True)
        else:
            st.info("No feedback recorded yet.")
