from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
from typing import List
from collections import defaultdict

_ = load_dotenv(find_dotenv())

#setup 
persist_directory = 'db/chroma_db'
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o", temperature=0)

db=Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

#pydantic model for the output 
class QueryVariations(BaseModel):
    queries: List[str]

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
    
#step 3: Apply Reciprocal Rank Fusion (RRF) to combine results from all query variations

def reciprocal_rank_fusion(chunk_lists, k=60, verbose=True):
    
    if verbose:
        print("\n" + "="*60)
        print("APPLYING RECIPROCAL RANK FUSION")
        print("="*60)
        print(f"\nUsing k={k}")
        print("Calculating RRF scores...\n")
        
    #Data structures for RRF calculation
    rrf_scores = defaultdict(float)  # Store RRF scores for each document
    all_unique_chunks = {}  # Will store : {chunk_content: actual chunk object} to avoid duplicates
    
    # For verbose output - track chunk IDs
    chunk_id_map = {}
    chunk_counter = 1
    
    # Go through each retrieval result
    for query_idx, chunks in enumerate(chunk_lists, 1):
        if verbose:
            print(f"Processing Query {query_idx} results:")
        
        # Go through each chunk in this query's results
        for position, chunk in enumerate(chunks, 1):  # position is 1-indexed
            # Use chunk content as unique identifier
            chunk_content = chunk.page_content
            
            # Assign a simple ID if we haven't seen this chunk before
            if chunk_content not in chunk_id_map:
                chunk_id_map[chunk_content] = f"Chunk_{chunk_counter}"
                chunk_counter += 1
            
            chunk_id = chunk_id_map[chunk_content]
            
            # Store the chunk object (in case we haven't seen it before)
            all_unique_chunks[chunk_content] = chunk
            
            # Calculate position score: 1/(k + position)
            position_score = 1 / (k + position)
            
            # Add to RRF score
            rrf_scores[chunk_content] += position_score
            
            if verbose:
                print(f"  Position {position}: {chunk_id} +{position_score:.4f} (running total: {rrf_scores[chunk_content]:.4f})")
                print(f"    Preview: {chunk_content[:80]}...")
        
        if verbose:
            print()
    
    # Sort chunks by RRF score (highest first)
    sorted_chunks = sorted(
        [(all_unique_chunks[chunk_content], score) for chunk_content, score in rrf_scores.items()],
        key=lambda x: x[1],  # Sort by RRF score
        reverse=True  # Highest scores first
    )
    
    if verbose:
        print(f"✅ RRF Complete! Processed {len(sorted_chunks)} unique chunks from {len(chunk_lists)} queries.")
    
    return sorted_chunks

# Apply RRF to our retrieval results
fused_results = reciprocal_rank_fusion(all_retrieval_results, k=60, verbose=True)

# ──────────────────────────────────────────────────────────────────
# Step 4: Display Final Fused Results
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL RRF RANKING")
print("="*60)

print(f"\nTop {min(5, len(fused_results))} documents after RRF fusion:\n")

for rank, (doc, rrf_score) in enumerate(fused_results[:5], 1):
    print(f"🏆 RANK {rank} (RRF Score: {rrf_score:.4f})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 50)

print(f"\n✅ RRF Complete! Fused {len(fused_results)} unique documents from {len(query_variations)} query variations.")
print("\n💡 Key benefits:")
print("   • Documents appearing in multiple queries get boosted scores")
print("   • Higher positions contribute more to the final score") 
print("   • Balanced fusion using k=60 for gentle position penalties")

# ──────────────────────────────────────────────────────────────────
# Optional: Quick Usage Examples
# ──────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("USAGE EXAMPLES")
print("="*60)