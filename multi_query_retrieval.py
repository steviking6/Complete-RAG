from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from typing import List

_ = load_dotenv(find_dotenv())

#setup
persistent_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

#Pydantic model for structured output 
class QueryVariations(BaseModel):
    queries:List[str]
    
# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

# Original query
original_query = "How does Tesla make money?"
print(f"Original Query: {original_query}\n")
print(f"Generating variations of the original query {original_query}...")

#step 1: Generate variations of the original query

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve revelant documents:

Original Query: {original_query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
query_variations = response.queries

print(f"Generated Query Variations:")
for i,variation in enumerate(query_variations):
    print(f"{i+1}. {variation}")

print("\n" + "=" * 50)

#step 2: Search with each Query variation Varaiations and store results 

retriever = db.as_retriever(search_kwargs={"k": 5})
all_retrieval_results = [] #strore for RRF calculation 

for i,query in enumerate(query_variations, 1):
    print(f"\n=== Results for query variation {i}: {query} ===")
    
    docs = retriever.invoke(query)
    all_retrieval_results.append(docs)
    
    print(f"Retrieved {len(docs)} documents for query variation {i}.")
    
    for j,docs in enumerate(docs, 1):
        print(f"\nDocument {j}:")
        print(f"Content: {docs.page_content[:200]}...")  # Print first 200 characters
        
    print("-" * 50)
    
print("\n" + "=" * 50)
print("Multi-query retrieval completed. All results stored for RRF calculation."
      )
    
