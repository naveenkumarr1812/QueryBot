import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chatbot_backend import (
    chatbot,
    generate_thread_title,
    ingest_pdf,
    thread_document_metadata,
    # Auth
    login_user,
    register_user,
    save_user_thread,
    get_user_threads,
    delete_user_thread,
)

st.set_page_config(page_title="QueryBot", page_icon="🤖", layout="wide")



# Auth page — shown when not logged in
def show_auth_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("QueryBot")
        st.caption("Your personal AI assistant with PDF, web search & calculator.")
        st.markdown("<br>", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["Login", "Create Account"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="your username")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    else:
                        ok, msg = login_user(username, password)
                        if ok:
                            _init_user_session(username.strip().lower())
                            st.rerun()
                        else:
                            st.error(msg)

        with tab_register:
            with st.form("register_form"):
                new_username = st.text_input("Choose a username", placeholder="e.g. john_doe")
                new_password = st.text_input(
                    "Choose a password", type="password", placeholder="min 6 characters"
                )
                confirm_password = st.text_input(
                    "Confirm password", type="password", placeholder="repeat password"
                )
                submitted = st.form_submit_button("Create Account", use_container_width=True)
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = register_user(new_username, new_password)
                        if ok:
                            st.success(msg + " Please log in.")
                        else:
                            st.error(msg)


def _init_user_session(username: str):
    """Set up session state for a freshly logged-in user."""
    st.session_state["username"] = username
    st.session_state["message_history"] = []
    st.session_state["ingested_docs"] = {}
    st.session_state["thread_titles"] = {}

    # Load this user's persisted threads from DB
    saved = get_user_threads(username)
    st.session_state["chat_threads"] = [t["thread_id"] for t in saved]
    for t in saved:
        st.session_state["thread_titles"][t["thread_id"]] = t["title"]

    # Start a fresh thread for this session
    new_tid = _make_thread_id(username)
    st.session_state["thread_id"] = new_tid
    _add_thread(new_tid)



# Utilities
def _make_thread_id(username: str) -> str:
    """Prefix thread IDs with username so they are globally unique per user."""
    return f"{username}:{uuid.uuid4()}"


def generate_thread_id() -> str:
    username = st.session_state.get("username", "anon")
    return _make_thread_id(username)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    _add_thread(thread_id)
    st.session_state["message_history"] = []
    # Persist immediately so it survives a new tab
    save_user_thread(st.session_state["username"], thread_id, "New Chat")


def _add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id: str) -> list:
    try:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        return state.values.get("messages", [])
    except Exception:
        return []


def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]



# Guard — show auth page if not logged in
if "username" not in st.session_state:
    show_auth_page()
    st.stop()



# Session state initialisation (only reached when logged in)
username: str = st.session_state["username"]

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# Always reload threads + titles from DB so they survive session restarts
saved = get_user_threads(username)
saved_ids = [t["thread_id"] for t in saved]

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = saved_ids

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

# Sync any titles that are missing or still showing as raw thread IDs
for t in saved:
    tid, title = t["thread_id"], t["title"]
    current = st.session_state["thread_titles"].get(tid, "")
    # Overwrite if missing, blank, "New Chat", or looks like a raw UUID/prefixed ID
    if not current or current == "New Chat" or current.startswith(username + ":"):
        st.session_state["thread_titles"][tid] = title

# Merge any DB threads not yet in session (e.g. opened from another tab)
for tid in saved_ids:
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

_add_thread(st.session_state["thread_id"])

thread_key: str = st.session_state["thread_id"]
thread_docs: dict = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads: list = st.session_state["chat_threads"][::-1]
selected_thread = None



# Sidebar
with st.sidebar:
    # User info + logout
    st.markdown(f"👤 **{username}**")
    if st.button("Logout", use_container_width=True):
        logout()
        st.rerun()

    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()

    # Document status
    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.success(
            f"📄 **{latest_doc.get('filename')}**\n\n"
            f"{latest_doc.get('chunks')} chunks · {latest_doc.get('documents')} page(s)"
        )
    else:
        st.info("No PDF uploaded yet for this chat.")

    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.info(f"`{uploaded_pdf.name}` is already indexed for this chat.")
        else:
            with st.status("Indexing PDF…", expanded=True) as status_box:
                try:
                    summary = ingest_pdf(
                        uploaded_pdf.getvalue(),
                        thread_id=thread_key,
                        filename=uploaded_pdf.name,
                    )
                    thread_docs[uploaded_pdf.name] = summary
                    status_box.update(label="✅ PDF indexed", state="complete", expanded=False)
                except RuntimeError as exc:
                    status_box.update(label="❌ Indexing failed", state="error", expanded=True)
                    st.error(f"**Error:** {exc}")
                except ValueError as exc:
                    status_box.update(label="❌ PDF error", state="error", expanded=True)
                    st.error(f"**PDF problem:** {exc}")
                except Exception as exc:
                    status_box.update(label="❌ Unexpected error", state="error", expanded=True)
                    st.error(f"**Unexpected error:** {exc}")

    st.divider()

    # Past conversations
    st.subheader("💬 Past Conversations")
    if not threads:
        st.caption("No past conversations yet.")
    else:
        for tid in threads:
            title = st.session_state["thread_titles"].get(tid, f"Chat {str(tid)[-8:]}…")
            is_active = tid == thread_key
            label = f"{'▶ ' if is_active else ''}{title}"
            col_title, col_del = st.columns([5, 1])
            with col_title:
                if st.button(label, key=f"thread-{tid}", use_container_width=True):
                    selected_thread = tid
            with col_del:
                if st.button("🗑", key=f"del-{tid}", help="Delete this chat"):
                    delete_user_thread(username, tid)
                    st.session_state["chat_threads"].remove(tid)
                    st.session_state["thread_titles"].pop(tid, None)
                    if tid == thread_key:
                        reset_chat()
                    st.rerun()



# Main chat area
st.title("QueryBot")
st.caption("Ask questions about your PDF, use web search, or calculate anything.")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document, search the web, or calculate…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate + persist title from the very first real message of this thread
    if st.session_state["thread_titles"].get(thread_key) in (None, "New Chat"):
        with st.spinner(""):
            title = generate_thread_title(user_input)
            st.session_state["thread_titles"][thread_key] = title
            save_user_thread(username, thread_key, title)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder: dict = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}`…", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}`…",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        try:
            ai_message = st.write_stream(ai_only_stream())
        except Exception as exc:
            st.error(f"Error generating response: {exc}")
            ai_message = f"[Error: {exc}]"

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message or ""}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 {doc_meta.get('filename')} — "
            f"{doc_meta.get('chunks')} chunks"
        )

st.divider()


# Load a past conversation when sidebar button is clicked
if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()