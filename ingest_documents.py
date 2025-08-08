import os
from typing import List, Union
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Configuration ---
# Directory where your input documents are stored.
# Create this folder and place your PDF, Word, and email files inside.
DOCUMENTS_DIRECTORY = "./source_documents"

# Chunking parameters for processing documents
CHUNK_SIZE = 700
CHUNK_OVERLAP = 70

def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Loads all supported documents from a specified directory using appropriate loaders.
    
    Args:
        directory_path (str): The path to the directory containing documents.
        
    Returns:
        List[Document]: A list of loaded Document objects.
    """
    all_documents = []
    
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return []

    # Map file extensions to their corresponding LangChain loaders
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".eml": UnstructuredEmailLoader
    }

    print(f"Scanning for supported documents in '{directory_path}'...")
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext in loaders:
            loader_class = loaders[file_ext]
            print(f"-> Loading file: {filename} with {loader_class.__name__}")
            try:
                loader = loader_class(file_path)
                documents = loader.load()
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"-> Skipping unsupported file: {filename}")

    print(f"\nSuccessfully loaded a total of {len(all_documents)} pages/documents.")
    return all_documents

def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Splits a list of documents into smaller, overlapping chunks.
    
    Args:
        documents (List[Document]): The list of documents to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        List[Document]: A new list of Document objects representing the chunks.
    """
    if not documents:
        return []
    
    print(f"Splitting documents into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully split into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # --- Instructions for use ---
    # 1. Create a folder named 'source_documents' in the same directory as this script.
    # 2. Place your .pdf, .docx, and .eml files inside this folder.
    # 3. Run this script.
    
    # Load all documents from the specified directory
    documents = load_documents_from_directory(DOCUMENTS_DIRECTORY)

    if documents:
        # Split the loaded documents into chunks
        document_chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

        # Optional: Print details of the first chunk to verify
        if document_chunks:
            print("\n--- Details of the first chunk ---")
            print(f"Content: {document_chunks[0].page_content[:200]}...")
            print(f"Metadata: {document_chunks[0].metadata}")
    else:
        print("No documents were loaded or split.")