from dotenv import load_dotenv, find_dotenv
from langchain_chroma import Chroma 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

#load env variables 
_ = load_dotenv(find_dotenv())

#connect to docs database 
persistent_directory = "db/chroma_db"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    embedding_function= embeddings,
    persist_directory= persistent_directory
)

#set up AI model 
model = ChatOpenAI(model="gpt-4o")

#store conversation as messages 
chat_history = []

def ask_question(user_question):
    print(f"\n---- You asked : {user_question} ----")
     
     #step 1 : Make the question clear using conversation history 
    if chat_history:
        messages = [
                SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question."),
        ]+ chat_history+ [
                HumanMessage(content=f"New question : {user_question}") 
        ]

        result = model.invoke(messages)
        search_question = result.content.strip()
        print(f"searching for : {search_question}")
    else:
        search_question = user_question
        
    #step 2 : Find revelant documents 
    retriver = db.as_retriever(search_kwargs={"k":3})
    docs  = retriver.invoke(search_question)
    
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")
        
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
    """
    
    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]
    
    result = model.invoke(messages)
    answer = result.content
    
    #step 5 : Remember this conversation 
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"Answer: {answer}")
    return answer 

#simple chat loop 

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() == 'quit':
            print("Goodbye!")
            break
            
        ask_question(question)

if __name__ == "__main__":
    start_chat()
        