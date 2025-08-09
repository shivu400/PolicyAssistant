import streamlit as st
import json
import sys
import io
from typing import List, Any
from contextlib import contextmanager
import requests

# 🚨 FIX: Put the SQLite version patch at the very top of the script
# before any libraries that depend on chromadb are imported.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Now, import the other libraries
from app_core import get_rag_components, get_langgraph_app, PolicyResponse
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import MultiQueryRetriever
from langchain_core.documents import Document
# --- Utility for capturing CLI output ---
@contextmanager
def st_stdout_redirect(placeholder):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        captured_output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        if captured_output:
            placeholder.code(captured_output, language='text')

# --- Streamlit App Configuration ---
st.set_page_config(page_title="📄 RAG Policy Assistant", layout="wide")

st.markdown(
    """
    <style>
    .st-emotion-cache-18j133g, .st-emotion-cache-1dp5vir {
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1dp5vir {
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-1dp5vir .st-emotion-cache-1f10g2y {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #333;
    }
    .st-emotion-cache-1dp5vir .st-emotion-cache-1f10g2y::before {
        content: "📄";
        margin-right: 0.5rem;
    }
    .st-emotion-cache-1dp5vir .st-markdown > p {
        font-size: 1rem;
        color: #666;
    }
    .st-emotion-cache-1DP8yQ {
        background-color: #e6f7ff;
        border-left: 5px solid #007bff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: none;
    }
    .st-emotion-cache-1DP8yQ .st-markdown p {
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Load Default RAG System (Cached) ---
@st.cache_resource(show_spinner=False)
def load_default_rag_system():
    try:
        with st.spinner("🚀 Initializing LLM connectivity..."):
            _, _, llm_model, _ = get_rag_components() 
            test_llm_response = llm_model.invoke("Hello, are you ready?")
            st.success(f"LLM connectivity test successful: {test_llm_response.content[:50]}...")
            
        with st.spinner("📚 Loading default RAG components..."):
            hf_embeddings, chroma_vector_store, llm_model, document_retriever = get_rag_components()
        with st.spinner("✨ Compiling Langgraph workflow..."):
            langgraph_app = get_langgraph_app(llm_model, document_retriever)
        st.success("Default RAG system loaded and ready!")
        return {
            "app": langgraph_app, 
            "llm": llm_model, 
            "embeddings": hf_embeddings, 
            "vector_store": chroma_vector_store
        }
    except requests.exceptions.ConnectionError as ce:
        st.error(f"CRITICAL: Initial LLM connectivity test failed with Connection Error. Please check your API key, network, and OpenRouter model status. Error: {ce}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load default RAG system: {e}. Please check your `app_core.py` and ensure the default ChromaDB is populated.")
        st.stop()

# --- Main App Logic ---

st.title("📄 Smart Policy Assistant")
st.markdown("Hello! I'm your AI assistant for health insurance policies.")

# Sidebar for file uploads
with st.sidebar:
    st.header("Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Choose your documents (PDF, DOCX, EML)", 
        type=["pdf", "docx", "eml"],
        accept_multiple_files=True
    )
    process_button = st.button("Process Documents")

    if 'uploaded_docs_processed' not in st.session_state:
        st.session_state.uploaded_docs_processed = False
    
    if process_button and uploaded_files:
        st.session_state.uploaded_docs_processed = False
        with st.spinner("Processing your documents..."):
            try:
                from app_core import process_uploaded_documents
                
                # Process the files to get chunks
                chunks = process_uploaded_documents(uploaded_files)
                
                # Get components from the default system
                default_system = load_default_rag_system()
                llm_model = default_system["llm"]
                hf_embeddings = default_system["embeddings"]
                
                # Create a new, temporary vector store for the uploaded documents
                uploaded_vector_store = Chroma.from_documents(
                    documents=chunks,
                    embedding=hf_embeddings
                )
                
                # Create a new retriever and app for this temporary vector store
                uploaded_retriever = uploaded_vector_store.as_retriever(search_kwargs={"k": 3})
                uploaded_app = get_langgraph_app(llm_model, uploaded_retriever)
                
                st.session_state.uploaded_app = uploaded_app
                st.session_state.uploaded_docs_processed = True
                st.success("Your documents have been processed! You can now ask questions about them.")
                
            except Exception as e:
                st.error(f"An error occurred while processing your files: {e}")
                st.session_state.uploaded_docs_processed = False
                
    st.markdown("---")
    if st.session_state.uploaded_docs_processed:
        st.info("Currently using your uploaded documents.")
    else:
        st.info("Using the default policy documents.")


# --- Determine which RAG app to use ---
if 'uploaded_app' in st.session_state and st.session_state.uploaded_docs_processed:
    langgraph_app = st.session_state.uploaded_app
else:
    langgraph_app = load_default_rag_system()["app"]


# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            try:
                response_data = json.loads(message.content)
                st.markdown(f"**Decision:** {response_data.get('decision', 'N/A')}")
                st.markdown(f"**Amount:** {response_data.get('amount', 'N/A')}")
                st.markdown(f"**Justification:** {response_data.get('justification', 'N/A')}")
                if response_data.get('referenced_clauses'):
                    st.markdown(f"**Referenced Clauses:** {', '.join(response_data['referenced_clauses'])}")
                else:
                    st.markdown("**Referenced Clauses:** None explicitly found.")
            except json.JSONDecodeError:
                st.markdown(message.content)

user_query = st.chat_input("Type your policy question here...")

if user_query:
    st.session_state.messages.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        cli_output_placeholder = st.empty()
        
        with st.spinner("Processing your query..."):
            try:
                converted_chat_history: List[BaseMessage] = [
                    msg for msg in st.session_state.messages
                ]
                
                with st_stdout_redirect(cli_output_placeholder):
                    final_state = langgraph_app.invoke(
                        {"question": user_query, "chat_history": converted_chat_history},
                        config={"recursion_limit": 100}
                    )

                response_answer_obj = final_state.get('answer', None)
                relevance_grade = final_state.get('relevance_grade')

                if isinstance(response_answer_obj, PolicyResponse):
                    st.markdown(f"**Decision:** {response_answer_obj.decision}")
                    st.markdown(f"**Amount:** {response_answer_obj.amount}")
                    st.markdown(f"**Justification:** {response_answer_obj.justification}")
                    if response_answer_obj.referenced_clauses:
                        st.markdown(f"**Referenced Clauses:** {', '.join(response_answer_obj.referenced_clauses)}")
                    else:
                        st.markdown("**Referenced Clauses:** None explicitly found.")
                    st.info(f"Relevance Grade for this query: {relevance_grade}")
                    st.session_state.messages.append(AIMessage(content=response_answer_obj.model_dump_json()))
                elif response_answer_obj is not None:
                    st.warning(response_answer_obj)
                    st.session_state.messages.append(AIMessage(content=response_answer_obj))
                else:
                    display_message = (
                        f"The system could not find relevant information in the provided policy documents to answer '{user_query}'. "
                        f"Relevance grade: {relevance_grade}. Consider rephrasing or checking if the information exists."
                    )
                    st.warning(display_message)
                    st.session_state.messages.append(AIMessage(content=display_message))

            except Exception as e:
                error_message_display = f"An unexpected error occurred during processing: {e}"
                st.error(error_message_display)
                st.session_state.messages.append(AIMessage(content=error_message_display))
                st.rerun()