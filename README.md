# QueryBot 🤖

I built QueryBot as a multi-tool AI assistant that can chat, search the web, solve calculations, and answer questions from uploaded PDFs.

## What This Project Does

- I implemented user authentication (register/login).
- I added multi-thread chat history per user.
- I integrated PDF upload with RAG (retrieval-augmented generation).
- I enabled DuckDuckGo web search as a tool.
- I added a calculator tool for arithmetic operations.
- I stream assistant responses in a Streamlit UI.

The backend uses LangGraph + LangChain tooling, and the frontend is built with Streamlit.

## Features

- Ask general questions and get LLM responses.
- Ask web-based questions and let the assistant use search.
- Ask math questions and let the assistant use the calculator.
- Upload a PDF and ask for summaries or specific document details.
- Create and revisit past conversation threads.

## Tech Stack

- Python
- Streamlit
- LangGraph
- LangChain + LangChain Community tools
- Groq LLM (openai/gpt-oss-120b configured in code)
- FAISS vector store
- FastEmbed embeddings (BAAI/bge-small-en-v1.5)
- PyMuPDF4LLM for PDF-to-markdown extraction
- SQLite for authentication and checkpoint persistence

## Project Structure

- chatbot_frontend.py: Streamlit app (auth, chat UI, thread handling)
- chatbot_backend.py: LangGraph flow, tools, PDF ingestion, auth logic
- requirements.txt: Python dependencies

## How I Designed the Flow

1. The user logs in or creates an account.
2. The user starts a chat thread.
3. Optionally, the user uploads a PDF.
4. The backend extracts text, chunks it, embeds it, and creates a FAISS retriever for that thread.
5. On each turn, LangGraph routes between:
   - chat_node for reasoning
   - tools for search/calculator/RAG calls
6. Conversation state is checkpointed in SQLite using thread_id.

## Prerequisites

- Python 3.10+
- Groq API key

## Setup

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Add environment variables

Create a .env file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Run

```bash
streamlit run chatbot_frontend.py
```

Then open the URL shown by Streamlit (usually http://localhost:8501).

## Usage

1. Create an account or log in.
2. Click New Chat to start a fresh thread.
3. Optionally upload a PDF from the sidebar.
4. Ask document, web, or math questions.
5. Reopen older threads from Past Conversations.

## Data and Persistence Notes

- I store user credentials and thread metadata in SQLite.
- I use LangGraph SQLite checkpointing to persist chat state.
- I keep PDF retrievers in memory during runtime.
- If the app restarts, PDF indexes reset and PDFs need to be re-uploaded.

## License

This project is licensed under the MIT License.

See [LICENCE](LICENCE) for full details.
