import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv 

from pathlib import Path 

load_dotenv()

def load_documents(docs_path: str | Path = "docs"):
    """load all exists files from the docs directory."""
    print(f"loading documents from {docs_path}...")
    
    #check if docs directory exists 
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"the directory {docs_path} do not exists. Please, create it and add your company files.")
    
    #load all txt files from the docs directory 
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8-sig"},
    )
    
    documents = loader.load()
    
    if len(documents) == 0:
        raise FileNotFoundError(f" No .txt files found in {docs_path}. Plz add your company documents ")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\n Document {i+1}:")
        print(f" Source : {doc.metadata['source']}")
        print(f" Content length : {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" metadata: {doc.metadata}")
    
    return documents

def  split_documents(documents, chunk_size=800, chunk_overlap=0):
    """split documents into smaller chunks with overlap."""
    
    print("Splitting docs into chunks...")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # if chunks:
    #     for i, chunk in enumerate(chunks[:5]):
    #         print(f"\n ------- Chunk {i+1} -------")
    #         print(f" Source : {chunk.metadata['source']}")
    #         print(f" Content length : {len(chunk.page_content)} characters")
    #         print(" Content : ...")
    #         print(chunk.page_content)
    #         print(f" metadata: {chunk.metadata}")
    #         print("-"*50)

    return chunks

def create_vectorstore(chunks, persist_directory="db/chroma_db"):
    """create and persist Chroma vectorstore from the document chunks."""
    
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    )
    
    #Create ChromaDB vecorstore from the document chunks 
    print("-----Creating vector store-----")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print("----Finished creating vector store-----")
    
    print(f"vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("main function")
    
    #1. Load the files
    documents = load_documents(docs_path="docs")
    
    #2. Chunking the files
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=0)
    
    #3. Embedding the chunks and storin into Vector DB
    vectorstore = create_vectorstore(chunks)
    
if __name__ == "__main__":
    main()
    
