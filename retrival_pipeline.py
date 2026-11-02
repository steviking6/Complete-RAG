import os 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

persistent_directory = "db/chroma_db"

#load embeddings and vector store 
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key = os.getenv("OPENAI_API_KEY"))

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

#searcher foir revelant information 
query = "which island does SpaceX lease for oits launches in the Pacific ?"

retriver = db.as_retriever(search_kwargs={"k":5})

# retriver = db.as_retriever(
#     search_type = "similarity_score_threshold",
#     search_kwargs = {
#         "k":5,
#         "score_threshold":0.3 #only return chunks with cosines similarity >= 0.3
#     }
# )

revelant_docs = retriver.invoke(query)

print(f"USER Query : {query}")
print("----Context----")
for i, doc in enumerate(revelant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
