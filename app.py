
import os
import streamlit as st
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from datetime import datetime

# ─── Documents ───
documents = [
    Document(page_content="""
    Beyondles is an AI company that builds automation systems for businesses.
    They specialize in n8n workflows, GPT-4o integrations, and multi-agent systems.
    Their main products include email automation, meeting summarizers, and project trackers.
    Founded in 2024, they serve startup founders who want to automate repetitive tasks.
    """, metadata={"source": "company_info"}),

    Document(page_content="""
    Beyondles pricing:
    - Basic automation package: $500 — single workflow
    - Multi-agent system: $1500 — up to 4 agents
    - Enterprise package: $3000 — unlimited workflows + support
    Payment accepted via Upwork, PayPal, and bank transfer.
    Delivery time: 3-7 business days per project.
    """, metadata={"source": "pricing"}),

    Document(page_content="""
    Beyondles support policy:
    - Free revisions for 14 days after delivery
    - Response time: within 24 hours
    - Support channels: Email and Slack
    - Maintenance packages available monthly
    - All workflows come with documentation
    """, metadata={"source": "support"})
]

# ─── Setup ───
@st.cache_resource
def setup_agent():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Qdrant
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="beyondles_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="beyondles_docs",
        embedding=embeddings
    )
    vectorstore.add_documents(chunks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # LLM
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2
    )

    # Tools
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search Beyondles knowledge base for company info, pricing, support."""
        docs = retriever.invoke(query)
        return "\n".join(doc.page_content for doc in docs)

    @tool
    def calculator(expression: str) -> str:
        """Calculate math expressions. Example: 1500 * 2"""
        try:
            return f"Result: {eval(expression)}"
        except:
            return "Invalid expression"

    @tool
    def get_current_date(query: str) -> str:
        """Get current date and time."""
        return f"Current date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    tools = [search_knowledge_base, calculator, get_current_date]

    # Agent
    memory = MemorySaver()
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="You are a helpful AI assistant for Beyondles AI company. Use tools to answer accurately. Always search knowledge base first for company-related questions.",
        checkpointer=memory
    )

    return agent

# ─── Streamlit UI ───
st.set_page_config(
    page_title="Beyondles AI Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Beyondles AI Agent")
st.caption("Powered by RAG + LangGraph + GPT-OSS 120B")
st.divider()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_session_001"

# Load agent
with st.spinner("Loading AI Agent..."):
    agent = setup_agent()

# Chat history dikhao
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input
if prompt := st.chat_input("Ask me anything about Beyondles..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.session_id}}
            result = agent.invoke(
                {"messages": [("human", prompt)]},
                config=config
            )
            response = result["messages"][-1].content
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
