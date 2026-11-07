"""Retrieval + génération multimodale.

On cherche sur les résumés, puis on renvoie au LLM le contenu ORIGINAL du chunk
(texte, tableaux HTML et images base64) pour qu'il réponde à partir des sources.
"""
import json
import sys

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import ANSWER_MODEL, COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR


def load_retriever(k: int = 3):
    db = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return db.as_retriever(search_kwargs={"k": k})


def build_multimodal_message(query: str, docs) -> HumanMessage:
    prompt = f"Based on the following documents, answer this question: {query}\n\nDOCUMENTS:\n"
    images = []

    for i, doc in enumerate(docs, 1):
        original = json.loads(doc.metadata["original_content"])
        prompt += f"\n--- Document {i} (source: {doc.metadata['source']}, page {doc.metadata['page_number']}) ---\n"
        prompt += f"TEXT:\n{original['text']}\n"
        for j, table in enumerate(original["tables"], 1):
            prompt += f"\nTABLE {j} (HTML):\n{table}\n"
        images.extend(original["images"])

    prompt += (
        "\nAnswer using the text, tables and images provided. "
        "If the documents don't contain the answer, say you don't have the information."
    )

    content = [{"type": "text", "text": prompt}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    return HumanMessage(content=content)


def answer(query: str, k: int = 3) -> str:
    docs = load_retriever(k).invoke(query)

    print(f"USER Query : {query}")
    print("---- Chunks récupérés ----")
    for i, doc in enumerate(docs, 1):
        print(
            f"{i}. {doc.metadata['source']} p.{doc.metadata['page_number']} "
            f"| tables={doc.metadata['has_tables']} images={doc.metadata['has_images']}"
        )

    llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0)
    response = llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant that answers from the provided documents."),
            build_multimodal_message(query, docs),
        ]
    )
    return response.content


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What BLEU score does the big Transformer get on English-to-German?"
    print("\n---- Réponse ----\n" + answer(question))
