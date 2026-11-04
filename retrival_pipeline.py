import os 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage 

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
query = "What was invidia first graphic accelerator ?"

# retriver = db.as_retriever(search_kwargs={"k":5})

retriver = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {
        "k":5,
        "score_threshold":0.3 #only return chunks with cosines similarity >= 0.3
    }
)

revelant_docs = retriver.invoke(query)

print(f"USER Query : {query}")
print("----Context----")
for i, doc in enumerate(revelant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

#combine the query and the revelant document contents 
combined_input = f"""Based on the following documents, please answer this question : {query}

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in revelant_docs])}

Please provide a clear, helpful answer using the information from these documents. If you can't find the answer in the document, just tell that you don't have informations to respond to the question

Find revelant information based on the provided documents, then answer the question based on the revelant information 

"""

#create ChatOpenAI model 
model = ChatOpenAI(model="gpt-4o")

#define the messages for the model 
messages = [
    SystemMessage(content="You are an helpful assistant."),
    HumanMessage(content=combined_input)
]

#Invoke the model with the combined input 
result = model.invoke(messages)

#display the full result and content only 
print("\n----Generated response-----")
#print("Full result:")
#print(result)
print("content only:")
print(result.content)
