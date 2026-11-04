
import streamlit as st
import hashlib
import uuid
import tempfile

from pathlib import Path

from app.vector_store import (
    process_pdfs,
    remove_pdf,
    cleanup_expired_collections,
    touch_collection,
    collection_exists,
    collection_has_documents,
    get_pdf_metadata
)

from app.rag_pipeline import answer_question

from app.chat_history import (
    init_db,
    save_message,
    get_messages
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PDF Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)


# ==========================================
# CUSTOM STYLING
# ==========================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #888888;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-heading {
        font-size: 21px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .info-card {
        padding: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        margin-bottom: 12px;
    }

    .status-text {
        color: #888888;
        font-size: 13px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stChatInput"] {
        border-radius: 14px;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.2);
        padding: 12px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">📚 PDF Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Upload your documents, ask questions, and get answers
        grounded in your PDFs.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# INITIALIZE DATABASE
# ==========================================

init_db()


# ==========================================
# UPLOAD LIMITS
# ==========================================

MAX_FILES = 3
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_TOTAL_SIZE = 25 * 1024 * 1024


# ==========================================
# SESSION ID
# ==========================================

session_id = st.query_params.get("sid")

try:

    if (
            not session_id
            or uuid.UUID(session_id).hex != session_id
    ):
        raise ValueError("Invalid session ID")

except (ValueError, AttributeError, TypeError):

    session_id = uuid.uuid4().hex
    st.query_params["sid"] = session_id


if "session_id" not in st.session_state:
    st.session_state.session_id = session_id


if "conversation_history" not in st.session_state:

    st.session_state.conversation_history = get_messages(
        session_id
    )


if "last_upload_signature" not in st.session_state:
    st.session_state.last_upload_signature = None


if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


if "pdf_details" not in st.session_state:
    st.session_state.pdf_details = []


collection_name = f"pdf_{session_id}"


# ==========================================
# AUTOMATIC CLEANUP
# ==========================================

if "cleanup_checked" not in st.session_state:

    cleanup_expired_collections(
        ttl_hours=24,
        exclude_collection=collection_name
    )

    st.session_state.cleanup_checked = True


# ==========================================
# RESTORE PDF DATA
# ==========================================

if collection_exists(collection_name):

    st.session_state.pdf_processed = (
        collection_has_documents(collection_name)
    )

    st.session_state.pdf_details = get_pdf_metadata(
        collection_name
    )

else:

    st.session_state.pdf_processed = False
    st.session_state.pdf_details = []


touch_collection(collection_name)


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📚 Knowledge Base")

    st.caption(
        "Manage your uploaded PDF documents."
    )

    st.divider()

    st.markdown("### 📤 Upload documents")

    st.caption(
        "Add up to 3 PDFs to your knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    # --------------------------------------
    # VALIDATE AND PROCESS
    # --------------------------------------

    if uploaded_files:

        total_size = sum(
            file.size for file in uploaded_files
        )

        if len(uploaded_files) > MAX_FILES:

            st.error(
                "Maximum 3 PDFs can be selected at a time."
            )

        elif any(
                file.size > MAX_FILE_SIZE
                for file in uploaded_files
        ):

            st.error(
                "Each PDF must not exceed 10 MB."
            )

        elif total_size > MAX_TOTAL_SIZE:

            st.error(
                "Combined upload size must not exceed 25 MB."
            )

        else:

            file_hashes = {}

            for file in uploaded_files:

                file_hash = hashlib.sha256(
                    file.getvalue()
                ).hexdigest()

                file_hashes[file.name] = file_hash

            file_signature = tuple(
                sorted(file_hashes.values())
            )

            if (
                    file_signature
                    != st.session_state.last_upload_signature
            ):

                existing_hashes = {
                    pdf["hash"]
                    for pdf in st.session_state.pdf_details
                }

                new_hashes = (
                        set(file_hashes.values()) - existing_hashes
                )

                current_count = len(
                    st.session_state.pdf_details
                )

                if current_count + len(new_hashes) > MAX_FILES:

                    st.error(
                        f"You already have {current_count} PDFs. "
                        f"You can add only "
                        f"{MAX_FILES - current_count} more."
                    )

                elif not new_hashes:

                    st.info(
                        "These PDFs are already uploaded."
                    )

                    st.session_state.last_upload_signature = (
                        file_signature
                    )

                else:

                    with st.spinner(
                            "Processing your PDFs..."
                    ):

                        try:

                            with tempfile.TemporaryDirectory(
                                    prefix="pdf_rag_"
                            ) as temp_dir:

                                upload_folder = Path(temp_dir)

                                pdf_paths = []

                                for file in uploaded_files:

                                    safe_name = Path(file.name).name

                                    pdf_path = (
                                            upload_folder / safe_name
                                    )

                                    with open(pdf_path, "wb") as f:
                                        f.write(file.getvalue())

                                    pdf_paths.append(pdf_path)

                                result = process_pdfs(
                                    pdf_paths,
                                    collection_name,
                                    replace_existing=False
                                )

                            st.session_state.pdf_processed = (
                                collection_has_documents(
                                    collection_name
                                )
                            )

                            st.session_state.pdf_details = (
                                get_pdf_metadata(collection_name)
                            )

                            st.session_state.last_upload_signature = (
                                file_signature
                            )

                            if result["success"]:

                                st.success(
                                    "PDFs processed successfully!"
                                )

                                st.caption(
                                    f"New chunks created: "
                                    f"{result['total_chunks']}"
                                )

                            elif result["skipped_pdfs"]:

                                st.info(
                                    "No new PDFs were added."
                                )

                            else:

                                st.error(result["message"])

                            if result["skipped_pdfs"]:

                                with st.expander(
                                        "Skipped PDFs"
                                ):

                                    for name, reason in result[
                                        "skipped_pdfs"
                                    ]:

                                        st.write(
                                            f"⚠️ {name} - {reason}"
                                        )

                        except Exception as e:

                            st.error(
                                f"PDF processing failed: {e}"
                            )

    st.divider()

    # --------------------------------------
    # DOCUMENT LIST
    # --------------------------------------

    st.markdown("### 📁 Your documents")

    if collection_exists(collection_name):

        st.session_state.pdf_details = get_pdf_metadata(
            collection_name
        )

        st.session_state.pdf_processed = (
            collection_has_documents(collection_name)
        )

    pdf_details = st.session_state.pdf_details

    if pdf_details:

        st.caption(
            f"{len(pdf_details)} / {MAX_FILES} documents uploaded"
        )

        for pdf in pdf_details:

            with st.container(border=True):

                st.markdown(
                    f"**📄 {pdf['name']}**"
                )

                st.caption(
                    f"{pdf['page_count']} pages · "
                    f"{pdf['chunk_count']} chunks"
                )

                st.caption(
                    f"Status: {pdf['status']}"
                )

                if st.button(
                        "Remove document",
                        key=f"remove_{pdf['hash']}",
                        use_container_width=True
                ):

                    with st.spinner("Removing document..."):

                        result = remove_pdf(
                            collection_name,
                            pdf["hash"]
                        )

                    if result["success"]:

                        st.session_state.pdf_details = (
                            result["pdf_details"]
                        )

                        st.session_state.pdf_processed = (
                            collection_has_documents(
                                collection_name
                            )
                        )

                        st.success(result["message"])

                        st.rerun()

                    else:

                        st.error(result["message"])

    else:

        st.info(
            "No documents uploaded yet."
        )

    st.divider()

    st.caption(
        "PDF Knowledge Assistant · Powered by RAG"
    )


# ==========================================
# MAIN CHAT AREA
# ==========================================

st.markdown(
    '<div class="section-heading">💬 Ask your documents</div>',
    unsafe_allow_html=True
)

if st.session_state.pdf_processed:

    st.success(
        f"{len(st.session_state.pdf_details)} PDF(s) ready "
        "for questions."
    )

else:

    st.info(
        "Upload a PDF using the sidebar to start chatting."
    )


# ==========================================
# CHAT HISTORY
# ==========================================

for message in st.session_state.conversation_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# CHAT INPUT
# ==========================================

question = st.chat_input(
    "Ask a question about your uploaded PDFs..."
)


if question:

    if not st.session_state.pdf_processed:

        st.warning(
            "Please upload valid PDFs first."
        )

    else:

        previous_history = (
            st.session_state.conversation_history.copy()
        )

        # ----------------------------------
        # USER MESSAGE
        # ----------------------------------

        user_message = {
            "role": "user",
            "content": question
        }

        st.session_state.conversation_history.append(
            user_message
        )

        save_message(
            session_id,
            "user",
            question
        )

        with st.chat_message("user"):

            st.markdown(question)

        # ----------------------------------
        # ASSISTANT RESPONSE
        # ----------------------------------

        with st.chat_message("assistant"):

            answer_placeholder = st.empty()

            with st.spinner(
                    "Searching your documents and generating an answer..."
            ):

                try:

                    answer, references = answer_question(
                        question,
                        collection_name,
                        previous_history
                    )

                    if not isinstance(answer, str):
                        answer = str(answer)

                    if not answer.strip():

                        answer = (
                            "Sorry, I could not generate an answer."
                        )

                    answer_placeholder.markdown(answer)

                    # --------------------------
                    # SOURCE REFERENCES
                    # --------------------------

                    if references:

                        st.divider()

                        st.markdown(
                            "**📚 Sources from your PDFs**"
                        )

                        for ref in references:

                            st.markdown(
                                f"- 📄 **{ref['source']}** "
                                f"— Page {ref['page']}"
                            )

                    # --------------------------
                    # SAVE ASSISTANT MESSAGE
                    # --------------------------

                    assistant_message = {
                        "role": "assistant",
                        "content": answer
                    }

                    st.session_state.conversation_history.append(
                        assistant_message
                    )

                    save_message(
                        session_id,
                        "assistant",
                        answer
                    )

                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )
