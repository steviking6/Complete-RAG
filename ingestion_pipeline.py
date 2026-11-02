import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv 

from pathlib import Path 

load_dotenv()

def load_documents(docs_path:Path = "docs"):
    """load all exists files from the docs directory."""
    print(f"loading documents from {docs_path}...")
    
    #check if docs directory exists 
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"the directory {docs_path} do not exists. Please, create it and add your company files.")
    
    #load all txt files from the docs directory 
    loader = DirectoryLoader(
        path=docs_path,
        glob=".txt",
        loader_cls=TextLoader
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

def main():
    print("main function")
    
if __name__ == "__main__":
    main()
    
