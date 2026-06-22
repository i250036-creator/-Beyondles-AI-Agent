# 🤖 Beyondles AI Agent

A production-ready AI agent built with RAG + LangGraph + GPT-OSS 120B

## 🔗 Live Demo
[beyondles-ai-agent.streamlit.app](https://beyondles-ai-agent.streamlit.app)

## 🛠 Tech Stack
- **LangGraph** — Agent brain + tool calling
- **Qdrant** — Vector store for semantic search
- **RAG Pipeline** — Document retrieval + AI answers
- **HuggingFace Embeddings** — sentence-transformers
- **OpenRouter** — GPT-OSS 120B LLM
- **Streamlit** — Live web interface

## ⚡ Features
- Multi-turn conversation memory
- RAG-based knowledge retrieval
- Tool calling (search, calculator, date)
- Live deployed web app

## 🏗 Architecture
User Question → Qdrant Vector Search → 
Relevant Docs → GPT-OSS 120B → Answer
