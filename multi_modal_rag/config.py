"""Configuration partagée entre l'ingestion et la retrieval."""
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
PERSIST_DIR = BASE_DIR / "db" / "chroma_db"
COLLECTION_NAME = "multimodal_rag"

EMBEDDING_MODEL = "text-embedding-3-small"
SUMMARY_MODEL = "gpt-4o-mini"  # résumés des chunks (images/tableaux) à l'ingestion
ANSWER_MODEL = "gpt-4o"  # génération de la réponse finale

# Binaires Windows installés via winget : unstructured en a besoin dans le PATH
# (tesseract pour l'OCR, poppler/pdftoppm pour convertir les pages PDF en images).
_WINDOWS_BINARIES = [
    Path(r"C:\Program Files\Tesseract-OCR"),
    *Path(os.getenv("LOCALAPPDATA", "")).glob(
        "Microsoft/WinGet/Packages/oschwartz10612.Poppler*/poppler-*/Library/bin"
    ),
]
for _bin in _WINDOWS_BINARIES:
    if _bin.exists() and str(_bin) not in os.environ["PATH"]:
        os.environ["PATH"] = str(_bin) + os.pathsep + os.environ["PATH"]
