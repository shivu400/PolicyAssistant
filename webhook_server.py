# File: webhook_server.py

import os
import json
import requests
import re
from flask import Flask, request, jsonify
from typing import List
from werkzeug.datastructures import FileStorage
import uuid
import io

# Import core RAG components from your backend module
from app_core import (
    get_rag_components,
    get_langgraph_app,
    PolicyResponse,
    get_huggingface_embeddings,
    process_uploaded_documents,
    GraphState
)

# A temporary, in-memory representation of a file object for our loader
class TemporaryFile:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content
        self.buffer = io.BytesIO(content)
        self.name = filename

    def getbuffer(self):
        return self.buffer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.buffer.close()

app = Flask(__name__)

# --- Load RAG Components Globally (on server startup) ---
try:
    print("Loading RAG components for webhook server...")
    hf_embeddings, chroma_vector_store, llm_model, document_retriever = get_rag_components()
    langgraph_app = get_langgraph_app(llm_model, document_retriever)
    print("RAG system loaded and ready.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load RAG system. Server will not function. Error: {e}")
    langgraph_app = None

# --- Configuration for Webhook ---
# Your team's token for authentication
BEARER_TOKEN = "c96a70a0e0ca2611043c5b543b5b6f5940d3bf5c480c5ac1236690b2f148783c"

def authenticate_token(token):
    return token == f"Bearer {BEARER_TOKEN}"

# --- Webhook Endpoint for HackRx Submissions ---
@app.route('/hackrx/run', methods=['POST'])
def run_submissions():
    # --- 1. Authentication Check ---
    auth_header = request.headers.get('Authorization')
    if not auth_header or not authenticate_token(auth_header):
        return jsonify({"error": "Unauthorized"}), 401
        
    # --- 2. Input Validation ---
    if not request.is_json:
        return jsonify({"error": "Invalid request. Expected JSON payload."}), 400

    payload = request.json
    document_url = payload.get("documents")
    questions = payload.get("questions")

    if not document_url or not questions or not isinstance(questions, list):
        return jsonify({"error": "Invalid payload format. 'documents' URL and 'questions' list are required."}), 400

    # --- 3. Document Download and Processing ---
    print(f"Downloading document from URL: {document_url}")
    try:
        response = requests.get(document_url)
        response.raise_for_status()
        
        filename = os.path.basename(document_url.split('?')[0])
        temp_file = TemporaryFile(filename, response.content)
        
        chunks = process_uploaded_documents([temp_file])

        temp_db_path = f"./temp_db_{uuid.uuid4()}"
        temp_vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=get_huggingface_embeddings("sentence-transformers/all-MiniLM-L6-v2"),
            persist_directory=temp_db_path
        )
        temp_vector_store.persist()
        
        temp_retriever = temp_vector_store.as_retriever(search_kwargs={"k": 3})
        temp_app = get_langgraph_app(llm_model, temp_retriever)
        
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to download document.", "message": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Failed to process document.", "message": str(e)}), 500
    
    # --- 4. Run RAG for each question ---
    answers = [] # FIX: Changed from 'results' to 'answers'
    for question in questions:
        print(f"Processing question: {question}")
        try:
            final_state = temp_app.invoke(
                {"question": question, "chat_history": []},
                config={"recursion_limit": 100}
            )
            
            response_answer_obj = final_state.get('answer', None)
            
            if isinstance(response_answer_obj, PolicyResponse):
                # FIX: Extract only the 'justification' from the structured response
                answers.append(response_answer_obj.justification) 
            else:
                answers.append("The system could not find a relevant answer for this question.") # FIX: Changed error message
        
        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            answers.append(f"An unexpected error occurred while processing this question: {str(e)}")

    # --- 5. Cleanup and Return Results ---
    if os.path.exists(temp_db_path):
        import shutil
        shutil.rmtree(temp_db_path)
        print(f"Cleaned up temporary database at {temp_db_path}")

    # FIX: Changed the final response format to match the required structure
    return jsonify({"answers": answers}), 200

if __name__ == '__main__':
    app.run(port=5000, debug=False)
