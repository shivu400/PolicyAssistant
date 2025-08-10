import os
import json
import requests
import re
from typing import List, Literal, TypedDict, Union, BinaryIO
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI as DeprecatedChatOpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field, PydanticDeprecatedSince20
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import MultiQueryRetriever

from dotenv import load_dotenv

# Langgraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, AnyMessage, HumanMessage, AIMessage

# Load environment variables from .env file (if present)
load_dotenv()

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
PRIMARY_LLM_MODEL_NAME = "pankajmathur/orca_mini_v3_7b"

# --- Pydantic Models for Structured Output ---
class PolicyResponse(BaseModel):
    decision: str = Field(description="The final decision regarding the query (e.g., 'Approved', 'Declined', 'Information Provided', 'Cannot Determine', 'Not Applicable').")
    amount: str = Field(description="The numerical amount relevant to the decision, if applicable (e.g., 'INR 50,000', 'USD 1,000', 'N/A'). If no specific amount is mentioned, use 'N/A'.")
    justification: str = Field(description="A detailed explanation for the decision or answer, directly referencing information from the policy documents. If the information is not found, state that.")
    referenced_clauses: List[str] = Field(description="A list of specific clause numbers (e.g., '1.', 'SECTION B)', 'PART A-I.1') or distinct section headings (e.g., 'Accident', 'Hospitalization', 'Exclusions- Standard') from the policy that directly support your justification. Extract these directly from the document text. If no specific clause is evident, use an empty list.")

# Pydantic Model for grading retrieved documents
class GradeDocuments(BaseModel):
    """Score for relevance of retrieved documents to the user question."""
    relevance_score: Literal["highly_relevant", "partially_relevant", "not_relevant"] = Field(
        description="Relevance score of retrieved documents to the question: 'highly_relevant', 'partially_relevant', or 'not_relevant'."
    )
    reason: str = Field(description="Brief reason for the assigned relevance score.")

# --- Langgraph: Define Graph State ---
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    answer: Union[str, PolicyResponse]
    relevance_grade: Literal["highly_relevant", "partially_relevant", "not_relevant"]
    chat_history: List[AnyMessage]


# --- Utility Function for Robust JSON Extraction ---
def extract_json_from_llm_output(text: str) -> str:
    """
    Attempts to extract a valid JSON string from text that might contain extraneous characters,
    markdown code blocks, or preambles. This version is more robust.
    """
    text = text.strip()

    # Strategy 1: Find content inside ```json ... ``` block
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        if (json_str.startswith('{') and json_str.endswith('}')) or \
           (json_str.startswith('[') and json_str.endswith(']')):
            return json_str

    # Strategy 2: Find the first '{' and the last '}' that form a valid JSON object
    try:
        decoder = json.JSONDecoder()
        json_obj, idx = decoder.raw_decode(text)
        return text[text.find('{') : idx].strip()
    except json.JSONDecodeError:
        pass

    # Strategy 3: Aggressive regex to find any {...} or [...] structure
    aggressive_match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
    if aggressive_match:
        return aggressive_match.group(0).strip()

    return text

# --- Document Processing Function for Streamlit and Webhook ---
def process_uploaded_documents(uploaded_files: List[BinaryIO]) -> List[Document]:
    """
    Loads and splits a list of uploaded files from Streamlit or a webhook.
    """
    all_documents = []
    supported_extensions = {
        "pdf": PyPDFLoader,
        "docx": UnstructuredWordDocumentLoader,
        "eml": UnstructuredEmailLoader
    }

    for uploaded_file in uploaded_files:
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            temp_file_path = f"./temp_{uuid.uuid4()}{file_extension}"
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            print(f"Loading document from temporary file: {uploaded_file.name}")

            if file_extension in supported_extensions:
                loader_class = supported_extensions[file_extension]
                loader = loader_class(temp_file_path)
                docs = loader.load()
                all_documents.extend(docs)
                print(f"Loaded {len(docs)} pages/documents from {uploaded_file.name}")
            else:
                print(f"Skipping unsupported file type: {uploaded_file.name}")

        except Exception as e:
            print(f"Error loading {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    if not all_documents:
        print("No supported documents were loaded.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=70,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Processed into {len(chunks)} chunks.")
    return chunks

def get_huggingface_embeddings(model_name: str):
    """Initializes and returns a HuggingFaceEmbeddings object."""
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

def load_chroma_db(db_path: str, embeddings):
    """Loads an existing Chroma vector store from disk."""
    if not os.path.exists(db_path):
        return None
    vector_store = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    return vector_store

def initialize_openrouter_llm(model_name: str, temperature: float = 0.1):
    """Initializes and returns a ChatOpenAI instance configured for OpenRouter with a specific model."""
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set.")

    print(f"Initializing LLM with OpenRouter model: {model_name} (Temp: {temperature})...")
    try:
        llm = ChatOpenAI(
            base_url=OPENROUTER_API_BASE,
            api_key=openrouter_api_key,
            model_name=model_name,
            temperature=temperature,
            request_timeout=60.0
        )
    except TypeError:
        llm = DeprecatedChatOpenAI(
            openai_api_base=OPENROUTER_API_BASE,
            openai_api_key=openrouter_api_key,
            model_name=model_name,
            temperature=temperature,
            request_timeout=60.0
        )
    print(f"LLM (OpenRouter - {model_name}) initialized.")
    return llm

def create_retriever(llm, vector_store, k_docs=3):
    """Creates a simple retriever from the loaded vector store."""
    if vector_store is None:
        print("No vector store provided. Cannot create a retriever.")
        return None
    print(f"Creating simple retriever to fetch top {k_docs} documents...")
    retriever = vector_store.as_retriever(search_kwargs={"k": k_docs})
    print("Simple retriever created.")
    return retriever

def get_rag_components(vector_store_path: str = CHROMA_DB_PATH):
    """Initializes and returns all RAG components."""
    hf_embeddings = get_huggingface_embeddings(HF_EMBEDDING_MODEL)
    chroma_vector_store = load_chroma_db(vector_store_path, hf_embeddings)

    llm_model = initialize_openrouter_llm(PRIMARY_LLM_MODEL_NAME, temperature=0.1)

    document_retriever = create_retriever(llm_model, chroma_vector_store, k_docs=3)
    return hf_embeddings, chroma_vector_store, llm_model, document_retriever

def retrieve_node(state: GraphState, retriever) -> GraphState:
    """Retrieves documents based on the user's question."""
    print("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    if retriever is None:
        print("No retriever available. Skipping retrieval.")
        return {"documents": [], "question": question, "chat_history": state.get("chat_history", [])}
    
    try:
        documents = retriever.invoke(question)
        print(f"Retrieved {len(documents)} documents.")
    except requests.exceptions.ConnectionError as ce:
        print(f"An API Connection Error occurred during retrieval: {ce}")
        print("This often indicates a temporary network issue or API rate limit/instability.")
        return {"documents": [], "question": question, "chat_history": state.get("chat_history", [])}
    except Exception as e:
        print(f"An unexpected error occurred during retrieval: {e}")
        return {"documents": [], "question": question, "chat_history": state.get("chat_history", [])}
    
    return {"documents": documents, "question": question, "chat_history": state.get("chat_history", [])}

def grade_documents_node(state: GraphState, llm) -> GraphState:
    """
    Grades the retrieved documents for relevance to the user's question.
    """
    print("---NODE: GRADE DOCUMENTS FOR RELEVANCE---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        print("No documents retrieved for grading. Marking as not relevant.")
        return {"relevance_grade": "not_relevant", "question": question, "documents": documents, "chat_history": state.get("chat_history", [])}

    parser = JsonOutputParser(pydantic_object=GradeDocuments)
    format_instructions = json.dumps(GradeDocuments.model_json_schema(), indent=2)

    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a document grader. Your sole purpose is to determine document relevance. "
                     "**CRITICAL: Your entire response MUST be a VALID JSON object.** "
                     "**DO NOT include any other text, preambles, or explanations outside the JSON.**\n\n"
                     "Your task is to assess the relevance of the provided 'documents' to the 'question'. "
                     "Consider if the documents contain direct answers, supporting information, or related concepts. "
                     "Output ONLY a JSON object with two keys: 'relevance_score' and 'reason'.\n\n"
                     "Possible 'relevance_score' values are: 'highly_relevant', 'partially_relevant', or 'not_relevant'.\n\n"
                     "JSON Schema:\n{format_instructions}"),
        ("human", "Question: {question}\n\nDocuments:\n{documents}\n\nAssess relevance. Respond ONLY with JSON.")
    ])

    docs_content = "\n\n".join([doc.page_content for doc in documents])
    raw_llm_output_message = None

    try:
        grade_chain = grade_prompt | llm
        raw_llm_output_message = grade_chain.invoke({
            "format_instructions": format_instructions,
            "question": question,
            "documents": docs_content
        })

        raw_llm_output = extract_json_from_llm_output(raw_llm_output_message.content)
        grade_response_dict = json.loads(raw_llm_output)
        grade_output = GradeDocuments(**grade_response_dict)
        relevance = grade_output.relevance_score
        reason = grade_output.reason

        print(f"Document Relevance Grade: {relevance} (Reason: {reason})")
        return {"relevance_grade": relevance, "question": question, "documents": documents, "chat_history": state.get("chat_history", [])}
    except requests.exceptions.ConnectionError as ce:
        print(f"An API Connection Error occurred during grading: {ce}")
        print("This often indicates a temporary network issue or API rate limit/instability.")
        return {"relevance_grade": "not_relevant", "question": question, "documents": documents, "chat_history": state.get("chat_history", [])}
    except Exception as e:
        print(f"Error grading documents: {e}.")
        if raw_llm_output_message:
            print(f"Failed to parse LLM output as JSON. Raw LLM output (AIMessage content):\n---\n{raw_llm_output_message.content}\n---")
        else:
            print("LLM did not return an output message.")
        print("Defaulting to not relevant. Please review LLM's JSON adherence.")
        return {"relevance_grade": "not_relevant", "question": question, "documents": documents, "chat_history": state.get("chat_history", [])}


def generate_node(state: GraphState, llm) -> GraphState:
    """
    Generates the final answer in a structured JSON format.
    """
    print("---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    chat_history = state["chat_history"]

    print(f"\nDEBUG: Generate Node - Current Question: '{question}'")
    print(f"DEBUG: Generate Node - Number of Documents in Context: {len(documents)}")
    if documents:
        print(f"DEBUG: Generate Node - First Document Content (snippet): {documents[0].page_content[:300]}...")
        print(f"DEBUG: Generate Node - First Document Metadata: {documents[0].metadata}")
    else:
        print("DEBUG: Generate Node - No documents provided to generator.")
    print(f"DEBUG: Generate Node - Full Chat History Length: {len(chat_history)}")
    if chat_history:
        for i, msg in enumerate(chat_history[-2:]):
            print(f"DEBUG: Generate Node - Chat History Message {len(chat_history)-2+i}: Role={msg.type}, Content='{msg.content[:150]}...'")

    parser = PydanticOutputParser(pydantic_object=PolicyResponse)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant specializing in health insurance policy documents. "
                     "Your primary goal is to answer the **CURRENT USER QUERY** by extracting and synthesizing information "
                     "**ONLY** from the provided 'Context'. If the exact information is not found in the 'Context', state that. "
                     "**CRITICAL: Your entire response MUST be a VALID JSON object.** "
                     "**DO NOT include any other text, preambles, or commentary outside the JSON.**\n\n"
                     "Fill all fields accurately based on the 'Context' and the 'CURRENT Question'.\n\n"
                     "For 'referenced_clauses', extract actual clause numbers (e.g., '1.', 'SECTION B)', 'PART A-I.1') or distinct section headings (e.g., 'Accident', 'Hospitalization', 'Exclusions- Standard') "
                     "directly from the provided 'Context' that support your justification. "
                     "If no specific clause is evident, use an empty list for 'referenced_clauses'. "
                     "If no specific amount is found, use 'N/A' for 'amount'.\n\n"
                     "JSON Schema:\n{format_instructions}\n\n"
                     "Context:\n{context}\n"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    context_str = "\n\n".join([doc.page_content for doc in documents])
    raw_llm_output_message = None

    try:
        generate_chain = prompt | llm
        raw_llm_output_message = generate_chain.invoke({
            "context": context_str,
            "input": question,
            "chat_history": chat_history,
            "format_instructions": format_instructions
        })
        
        raw_llm_output = extract_json_from_llm_output(raw_llm_output_message.content)
        response_object_dict = json.loads(raw_llm_output)
        response_object = PolicyResponse(**response_object_dict)

        print(f"Generated Structured Answer:\n{json.dumps(response_object.model_dump(), indent=2)}")

        updated_chat_history = chat_history + [
            HumanMessage(content=question),
            AIMessage(content=json.dumps(response_object.model_dump()))
        ]

        return {"answer": response_object, "chat_history": updated_chat_history}

    except requests.exceptions.ConnectionError as ce:
        print(f"An API Connection Error occurred during generation: {ce}")
        print("This often indicates a temporary network issue or API rate limit/instability.")
        fallback_response_obj = PolicyResponse(
            decision="Cannot Determine",
            amount="N/A",
            justification=f"I encountered an API connection error trying to generate a response for the query: '{question}'. This might be a temporary issue with the language model service. Please try again in a moment.",
            referenced_clauses=[]
        )
        updated_chat_history = chat_history + [HumanMessage(content=question), AIMessage(content=json.dumps(fallback_response_obj.model_dump()))]
        return {"answer": fallback_response_obj, "chat_history": updated_chat_history}
    except Exception as e:
        print(f"Error generating structured answer: {e}.")
        if raw_llm_output_message:
            print(f"Failed to parse LLM output as JSON. Raw LLM output (AIMessage content):\n---\n{raw_llm_output_message.content}\n---")
        else:
            print("LLM did not return an output message.")
        fallback_response_obj = PolicyResponse(
            decision="Cannot Determine",
            amount="N/A",
            justification=f"I encountered an error trying to generate a structured response for the query: '{question}'. The precise information might be missing or the model failed to format correctly. Original error: {e}. Please try rephrasing your question.",
            referenced_clauses=[]
        )
        updated_chat_history = chat_history + [HumanMessage(content=question), AIMessage(content=json.dumps(fallback_response_obj.model_dump()))]
        return {"answer": fallback_response_obj, "chat_history": updated_chat_history}


# --- Langgraph: Define Conditional Edge (Router) ---
def route_documents(state: GraphState) -> Literal["generate", "end_no_info"]:
    """
    Conditional router to decide if an answer can be generated or if
    no relevant information was found, based on nuanced grades.
    """
    print("---NODE: ROUTING BASED ON DOCUMENT RELEVANCE---")
    relevance_grade = state["relevance_grade"]

    if relevance_grade in ["highly_relevant", "partially_relevant"]:
        print("Documents are deemed relevant, proceeding to GENERATE.")
        return "generate"
    else:
        print("Documents are not relevant enough, ending process (no info found).")
        return "end_no_info"

# --- Function to compile the Langgraph app ---
def get_langgraph_app(llm_model, document_retriever):
    """Builds and compiles the Langgraph workflow."""
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", lambda state: retrieve_node(state, document_retriever))
    workflow.add_node("grade_documents", lambda state: grade_documents_node(state, llm_model))
    workflow.add_node("generate", lambda state: generate_node(state, llm_model))

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_documents,
        {
            "generate": "generate",
            "end_no_info": END
        }
    )
    workflow.add_edge("generate", END)

    app = workflow.compile()
    return app
