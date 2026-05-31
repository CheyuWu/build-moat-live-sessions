import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_PATH = Path(__file__).resolve().parents[3] / ".kb" / "index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }


sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0
_index_lock = threading.RLock()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
    result: list[Section] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current_heading = ""
    current_content: list[str] = []

    def flush_section() -> None:
        if not current_heading:
            return
        content = "\n".join(current_content).strip()
        heading_path = [h for _, h in heading_stack]
        section_id = f"{path.name}#{slugify(current_heading)}"

        tokens = tokenize(" ".join(heading_path + [content]))
        result.append(
            Section(
                id=section_id,
                file=path.name,
                heading=current_heading,
                heading_path=heading_path,
                content=content,
                tokens=tokens,
            )
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        m = HEADING_RE.match(line)
        if m:
            flush_section()
            level, heading = len(m.group(1)), m.group(2)

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))
            current_heading = heading
            current_content = []
        else:
            if line.strip():
                current_content.append(line)

    flush_section()
    return result


def write_index_json(index_path: Path = INDEX_PATH) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sections": [s.to_dict() for s in sections],
        "stats": {
            "files": files_indexed,
            "sections": len(sections),
            "avg_doc_len": avg_doc_len,
        },
    }
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def rebuild_stats() -> None:
    global doc_freq, avg_doc_len, files_indexed
    doc_freq.clear() 

    files_indexed = len({s.file for s in sections})
    for section in sections:
        for token in set(section.tokens):
            doc_freq[token] += 1
    avg_doc_len = sum(len(s.tokens) for s in sections) / len(sections) if sections else 0.0


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
    global sections
    if not index_path.exists():
        return 0, 0
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    with _index_lock:
        sections = [Section(**item) for item in payload["sections"]]
        rebuild_stats()
        return files_indexed, len(sections)


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    global sections, doc_freq, avg_doc_len, files_indexed

    new_sections = []
    for md_path in sorted(docs_dir.glob("*.md")):
        new_sections.extend(parse_markdown(md_path))

    with _index_lock:
        sections = new_sections
        doc_freq = Counter()
        avg_doc_len = 0.0
        files_indexed = 0
        rebuild_stats()
        write_index_json()
    return files_indexed, len(sections)


def bm25_score(
    query_tokens: list[str],
    section: Section,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    if not sections or avg_doc_len == 0.0:
        return 0.0
    tf_map = Counter(section.tokens)
    doc_len = len(section.tokens)
    n = len(sections)
    heading_tokens = set(tokenize(" ".join(section.heading_path)))
    score = 0.0
    for term in query_tokens:
        tf = tf_map[term]
        if tf == 0:
            continue
        df = doc_freq[term]
        idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
        term_score = tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len)) * idf
        if term in heading_tokens:
            term_score *= 1.1
        score += term_score
    return score


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    with _index_lock:
        ranked = [(section, bm25_score(query_tokens, section)) for section in sections]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
