"""Ingestion multimodale de PDF avec unstructured.io.

1. partition_pdf (stratégie hi_res) : détection de layout + OCR, extraction des
   tableaux (HTML) et des images (base64).
2. chunk_by_title : regroupe les éléments par section.
3. Pour chaque chunk contenant un tableau ou une image, GPT génère un résumé
   textuel cherchable. C'est ce résumé qui est embeddé.
4. Le contenu original (texte, HTML des tableaux, images) est stocké dans les
   métadonnées pour être renvoyé au LLM au moment de la génération.
"""
import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from config import (
    COLLECTION_NAME,
    DOCS_DIR,
    EMBEDDING_MODEL,
    PERSIST_DIR,
    SUMMARY_MODEL,
)


def partition_document(file_path: Path):
    """Découpe le PDF en éléments typés (Title, NarrativeText, Table, Image...)."""
    print(f"Partitioning {file_path.name} (hi_res, peut prendre quelques minutes)...")
    elements = partition_pdf(
        filename=str(file_path),
        strategy="hi_res",  # modèle de layout : seule stratégie qui extrait les tableaux
        infer_table_structure=True,  # tableaux conservés en HTML dans metadata.text_as_html
        extract_image_block_types=["Image", "Table"],  # crops des images ET des tableaux
        extract_image_block_to_payload=True,  # crops en base64 dans metadata.image_base64
        languages=["eng"],
    )
    counts = {}
    for el in elements:
        counts[el.category] = counts.get(el.category, 0) + 1
    print(f"  {len(elements)} elements : {counts}")
    return elements


def create_chunks(elements):
    """Regroupe les éléments par section (titre) en chunks."""
    chunks = chunk_by_title(
        elements,
        max_characters=3000,  # taille max d'un chunk
        new_after_n_chars=2400,  # on coupe dès qu'on dépasse ce seuil
        combine_text_under_n_chars=500,  # fusionne les petites sections
    )
    print(f"  {len(chunks)} chunks créés")
    return chunks


def separate_content_types(chunk):
    """Sépare texte, tableaux (HTML) et images (base64) d'un chunk."""
    orig_elements = getattr(chunk.metadata, "orig_elements", None)
    if not orig_elements:
        # Pas d'éléments d'origine : on ne peut rien séparer, on garde le texte fusionné
        return {"text": chunk.text, "tables": [], "images": []}

    text_parts, tables, images = [], [], []
    for el in orig_elements:
        if el.category == "Table":
            # Le tableau va dans "tables" (HTML) et PAS dans le texte,
            # sinon il serait envoyé deux fois au LLM (aplati + HTML).
            tables.append(el.metadata.text_as_html or el.text)
            # Capture du tableau (dispo grâce à extract_image_block_types=["Table"]) :
            # filet de sécurité si le HTML est mal reconstruit (cellules fusionnées...).
            if el.metadata.image_base64:
                images.append(el.metadata.image_base64)
        else:
            if el.category == "Image" and el.metadata.image_base64:
                images.append(el.metadata.image_base64)
            if el.text:
                text_parts.append(el.text)  # inclut le texte OCR des images

    return {"text": "\n\n".join(text_parts), "tables": tables, "images": images}


def summarise_chunk(content, llm) -> str:
    """Résumé cherchable d'un chunk multimodal (texte + tableaux + images)."""
    prompt = f"""You are creating a searchable description for document retrieval.

TEXT CONTENT:
{content["text"]}
"""
    for i, table in enumerate(content["tables"], 1):
        prompt += f"\nTABLE {i} (HTML):\n{table}\n"

    prompt += """
Write a detailed, searchable description covering:
1. Key facts, numbers and data points from the text and tables
2. What the images/figures show and what they mean
3. The questions a user could ask that this content answers
4. Alternative terms and synonyms

SEARCHABLE DESCRIPTION:"""

    message_content = [{"type": "text", "text": prompt}]
    for img in content["images"]:
        message_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        )

    try:
        response = llm.invoke([HumanMessage(content=message_content)])
        return response.content
    except Exception as e:  # on ne bloque pas toute l'ingestion pour un chunk
        print(f"  Résumé IA échoué ({e}), fallback sur le texte brut")
        return content["text"]


def build_documents(chunks, source: str):
    """Transforme les chunks unstructured en Documents LangChain."""
    llm = ChatOpenAI(model=SUMMARY_MODEL, temperature=0)
    documents = []

    for i, chunk in enumerate(chunks, 1):
        content = separate_content_types(chunk)
        is_multimodal = bool(content["tables"] or content["images"])
        print(
            f"  Chunk {i}/{len(chunks)} : {len(content['tables'])} table(s), "
            f"{len(content['images'])} image(s)"
        )

        # Seuls les chunks avec tableau/image passent par le LLM (coût maîtrisé)
        page_content = summarise_chunk(content, llm) if is_multimodal else content["text"]

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": source,
                    "page_number": chunk.metadata.page_number or -1,
                    "has_tables": bool(content["tables"]),
                    "has_images": bool(content["images"]),
                    # Chroma n'accepte que des scalaires : on sérialise en JSON
                    "original_content": json.dumps(content),
                },
            )
        )
    return documents


def create_vector_store(documents):
    print(f"Embedding de {len(documents)} documents dans {PERSIST_DIR}...")
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)  # ré-ingestion propre, évite les doublons
    return Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )


def main():
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {DOCS_DIR}")

    documents = []
    for pdf in pdf_files:
        elements = partition_document(pdf)
        chunks = create_chunks(elements)
        documents.extend(build_documents(chunks, source=pdf.name))

    create_vector_store(documents)
    print("Ingestion terminée.")


if __name__ == "__main__":
    main()
