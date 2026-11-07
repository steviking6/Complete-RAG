from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

persistent_dir = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
#setup

db = Chroma(
    persist_directory= persistent_dir,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"} 
)

#query to test
query = "How much Microsoft pay to acquire Github ?"
print(f"Qery : {query}")

#METHOD 1 : Basic similarity search 
#Return top k similarity keywords 

print("======Method 1 : Similarity Search (k=3)====")
retriever = db.as_retriever(seach_kwargs={"k":3})

docs = retriever.invoke(query)
print(f"retrieved {len(docs)} documents")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")

print("-" * 60)

# ──────────────────────────────────────────────────────────────────
# METHOD 2: Similarity with Score Threshold
# Only returns documents above a certain similarity score
# ──────────────────────────────────────────────────────────────────

# print("\n=== METHOD 2: Similarity with Score Threshold ===")
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k": 3,
#         "score_threshold": 0.3  # Only return docs with similarity >= 0.3
#     }
# )

# docs = retriever.invoke(query)
# print(f"Retrieved {len(docs)} documents (threshold: 0.3):\n")

# for i, doc in enumerate(docs, 1):
#     print(f"Document {i}:")
#     print(f"{doc.page_content}\n")

# print("-" * 60)

# # ──────────────────────────────────────────────────────────────────
# # METHOD 3: Maximum Marginal Relevance (MMR)
# # Balances relevance and diversity - avoids redundant results
# # ──────────────────────────────────────────────────────────────────

# print("\n=== METHOD 3: Maximum Marginal Relevance (MMR) ===")
# retriever = db.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 3,           # Final number of docs
#         "fetch_k": 10,    # Initial pool to select from
#         "lambda_mult": 0.5  # 0=max diversity, 1=max relevance
#     }
# )

# docs = retriever.invoke(query)
# print(f"Retrieved {len(docs)} documents (λ=0.5):\n")

# for i, doc in enumerate(docs, 1):
#     print(f"Document {i}:")
#     print(f"{doc.page_content}\n")

# print("=" * 60)
# print("Done! Try different queries or parameters to see the differences.")