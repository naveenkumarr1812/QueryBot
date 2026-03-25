from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import pymupdf4llm
from langchain_core.documents import Document

load_dotenv()



# Auth — users table in the same SQLite DB
_AUTH_DB_PATH = "/tmp/chatbot_auth.db"

def _get_auth_conn():
    conn = sqlite3.connect(_AUTH_DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username  TEXT PRIMARY KEY,
            password  TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_threads (
            username  TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            title     TEXT NOT NULL DEFAULT 'New Chat',
            PRIMARY KEY (username, thread_id)
        )
        """
    )
    conn.commit()
    return conn


def _hash_password(password: str) -> str:
    import hashlib, os
    salt = os.urandom(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
    return salt.hex() + ":" + hashed.hex()


def _verify_password(password: str, stored: str) -> bool:
    import hashlib
    try:
        salt_hex, hash_hex = stored.split(":")
        salt = bytes.fromhex(salt_hex)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
        return hashed.hex() == hash_hex
    except Exception:
        return False


def register_user(username: str, password: str) -> Tuple[bool, str]:
    """Returns (success, message)."""
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    conn = _get_auth_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, _hash_password(password)),
        )
        conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."
    finally:
        conn.close()


def login_user(username: str, password: str) -> Tuple[bool, str]:
    """Returns (success, message)."""
    username = username.strip().lower()
    conn = _get_auth_conn()
    try:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            return False, "Username not found."
        if not _verify_password(password, row[0]):
            return False, "Incorrect password."
        return True, "Login successful."
    finally:
        conn.close()


def save_user_thread(username: str, thread_id: str, title: str = "New Chat") -> None:
    conn = _get_auth_conn()
    try:
        conn.execute(
            """
            INSERT INTO user_threads (username, thread_id, title)
            VALUES (?, ?, ?)
            ON CONFLICT(username, thread_id) DO UPDATE SET title=excluded.title
            """,
            (username, thread_id, title),
        )
        conn.commit()
    finally:
        conn.close()


def get_user_threads(username: str) -> List[dict]:
    """Return list of {thread_id, title} for a user, newest first."""
    conn = _get_auth_conn()
    try:
        rows = conn.execute(
            "SELECT thread_id, title FROM user_threads WHERE username = ? ORDER BY rowid DESC",
            (username,),
        ).fetchall()
        return [{"thread_id": r[0], "title": r[1]} for r in rows]
    finally:
        conn.close()


def delete_user_thread(username: str, thread_id: str) -> None:
    conn = _get_auth_conn()
    try:
        conn.execute(
            "DELETE FROM user_threads WHERE username = ? AND thread_id = ?",
            (username, thread_id),
        )
        conn.commit()
    finally:
        conn.close()


# 1. LLM
llm = ChatGroq(model="openai/gpt-oss-120b")


# 2. Embeddings — FastEmbed runs fully locally, no API call needed.
embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    cache_dir="/tmp/fastembed_cache",
)


# 3. In-memory retriever store (per session / thread)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        md_text = pymupdf4llm.to_markdown(temp_path)

        if not md_text.strip():
            raise ValueError(
                "No text could be extracted from the PDF. "
                "It may be corrupted or contain only images."
            )

        docs = [Document(page_content=md_text, metadata={"source": filename or temp_path})]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("Text was found but could not be split into chunks.")

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    except Exception as exc:
        raise RuntimeError(f"PDF ingestion failed: {exc}") from exc
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass



# 4. Thread title generator
def generate_thread_title(first_user_message: str) -> str:
    """Ask the LLM to summarise the first message in 4-5 words."""
    try:
        response = llm.invoke([
            SystemMessage(
                content=(
                    "You are a title generator. "
                    "Given a user message, reply with ONLY a 4-5 word title "
                    "that summarises the topic. "
                    "No punctuation, no quotes, no extra text."
                )
            ),
            HumanMessage(content=first_user_message),
        ])
        title = response.content.strip().strip('"').strip("'")
        if not title or len(title) > 60:
            return first_user_message[:40]
        return title
    except Exception:
        return first_user_message[:40]



# 5. Tools
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        ops = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b if b != 0 else None,
        }
        if operation not in ops:
            return {"error": f"Unsupported operation '{operation}'"}
        if operation == "div" and second_num == 0:
            return {"error": "Division by zero is not allowed"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": ops[operation](first_num, second_num),
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Search and retrieve content from the PDF document uploaded by the user.

    Use this tool for ANY of the following:
    - Questions about the document contents
    - Requests for a summary, overview, or brief of the document
    - "What is this document about?", "Tell me about the PDF", "Summarise the document"
    - Specific questions whose answer may be in the document
    - Any mention of "the document", "the PDF", "the file", "the uploaded file"

    For summary/overview requests, call this tool multiple times with different
    queries like "document overview", "main topics", "key points" to gather
    enough context before summarising.

    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Please upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "metadata": [doc.metadata for doc in result],
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [search_tool, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)


# 6. State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 7. Nodes
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    # Check if a document is available for this thread
    doc_available = thread_id and _get_retriever(thread_id) is not None
    doc_context = (
        f"A PDF document is currently uploaded and indexed for thread `{thread_id}`. "
        "You MUST use the `rag_tool` whenever the user asks anything about the document, "
        "including summaries, overviews, briefs, or specific questions. "
        "For summary requests, call `rag_tool` with queries like "
        "'document overview', 'main topics', and 'key points' to collect enough context, "
        "then provide a concise summary based on the retrieved content."
        if doc_available
        else
        "No document is uploaded yet for this thread. "
        "If the user asks about a document or PDF, politely ask them to upload one using the sidebar."
    )

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant with access to web search, a calculator, "
            "and a document retrieval tool (rag_tool).\n\n"
            f"{doc_context}\n\n"
            "General rules:\n"
            "- Always use the most appropriate tool for the user's question.\n"
            "- For web questions, use the search tool.\n"
            "- For math, use the calculator tool.\n"
            "- For anything about the uploaded document, use the rag_tool.\n"
            f"- Always pass thread_id=`{thread_id}` when calling rag_tool."
        )
    )

    response = llm_with_tools.invoke([system_message, *state["messages"]], config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)


# 8. Checkpointer
_DB_PATH = "/tmp/chatbot.db"
conn = sqlite3.connect(database=_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)


# 9. Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)



# 10. Helpers
def retrieve_all_threads() -> list:
    """Kept for backwards compat — not used when auth is enabled."""
    try:
        return list({
            cp.config["configurable"]["thread_id"]
            for cp in checkpointer.list(None)
        })
    except Exception:
        return []


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})