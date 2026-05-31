import json
import os
import re
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[3] / ".kb" / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "],
)

vectorstore: FAISS | None = None
_embeddings = None
files_indexed = 0
sections_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def get_embeddings():
    global _embeddings
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the server environment")
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            request_timeout=20,
            max_retries=1,
        )
    return _embeddings


def load_markdown_sections(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    filename = path.name

    docs = []
    current_heading = filename.replace(".md", "")
    current_lines: list[str] = []

    def flush(heading: str, lines: list[str]) -> None:
        content = "\n".join(lines).strip()
        if not content:
            return
        docs.append(
            Document(
                page_content=f"{heading}\n\n{content}",
                metadata={
                    "source": f"{filename}#{slugify(heading)}",
                    "heading": heading,
                },
            )
        )

    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush(current_heading, current_lines)
            current_heading = m.group(2)
            current_lines = []
        else:
            current_lines.append(line)

    flush(current_heading, current_lines)
    return docs


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    all_docs: list[Document] = []
    count_files = 0

    for md_file in sorted(docs_dir.glob("*.md")):
        sections = load_markdown_sections(md_file)
        all_docs.extend(sections)
        count_files += 1

    chunks = splitter.split_documents(all_docs)

    vectorstore = FAISS.from_documents(chunks, get_embeddings())

    files_indexed = count_files
    sections_indexed = len(chunks)

    save_vector_index()
    return files_indexed, sections_indexed


def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    if vectorstore is None:
        if index_dir.exists():
            shutil.rmtree(index_dir)
        return

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))

    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "sections_indexed": sections_indexed,
    }
    with (index_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed

    if (
        not (index_dir / "index.faiss").exists()
        or not (index_dir / "index.pkl").exists()
    ):
        return 0, 0

    meta_path = index_dir / "metadata.json"
    if meta_path.exists():
        meta: dict = json.loads(meta_path.read_text())

        # TODO: 確認儲存的 embedding model 和現在的 EMBEDDING_MODEL 一樣
        #
        # 如果不一樣，代表 index 是用不同 model 建的，直接 return (0, 0)
        # 提示：比較 meta.get("embedding_model") 和 EMBEDDING_MODEL
        if meta.get("embedding_model") != EMBEDDING_MODEL:
            return 0, 0
        
        files_indexed = meta.get("files_indexed", 0)
        sections_indexed = meta.get("sections_indexed", 0)

    # TODO: 載入 FAISS index 到全域 vectorstore
    #
    # 提示：FAISS.load_local(str(index_dir), get_embeddings(), allow_dangerous_deserialization=True)
    vectorstore = FAISS.load_local(str(index_dir), get_embeddings(), allow_dangerous_deserialization=True)
    return files_indexed, sections_indexed


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
