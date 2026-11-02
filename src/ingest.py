import hashlib
import json
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common import load_config, get_embeddings, load_store


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"-\n(\w)", r"\1", text)          # fix hyphenated line breaks
    text = re.sub(r"(?m)^\s*\d+\s*$", "", text)     # drop bare page numbers
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_file(path: Path):
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [Document(page_content=p.extract_text() or "",
                         metadata={"source": path.name, "page": i})
                for i, p in enumerate(reader.pages, 1)]
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name, "page": 1})]


def file_hash(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main():
    cfg = load_config()
    raw = Path(cfg["paths"]["raw"])
    index_dir = Path(cfg["paths"]["index"])
    manifest_path = Path(cfg["paths"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    emb = get_embeddings(cfg)
    vs = load_store(cfg)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunking"]["size"],
        chunk_overlap=cfg["chunking"]["overlap"],
    )
    stats = dict(files=0, skipped=0, pages=0, chunks=0, duplicates=0, empty=0)

    for path in sorted(raw.glob("*")):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        h = file_hash(path)
        if manifest.get(path.name) == h:            # unchanged file -> skip
            stats["skipped"] += 1
            continue

        if vs is not None:                          # remove the old version of this file
            old = [i for i, d in vs.docstore._dict.items()
                   if d.metadata.get("source") == path.name]
            if old:
                vs.delete(old)

        pages = load_file(path)
        stats["files"] += 1
        stats["pages"] += len(pages)

        existing = set(vs.docstore._dict) if vs is not None else set()
        new, seen = [], set()
        for page in pages:
            page.page_content = clean(page.page_content)
            for c in splitter.split_documents([page]):
                text = c.page_content.strip()
                if len(text) < 30:
                    stats["empty"] += 1
                    continue
                cid = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
                if cid in seen or cid in existing:  # exact-duplicate removal
                    stats["duplicates"] += 1
                    continue
                seen.add(cid)
                c.metadata["chunk_id"] = cid
                new.append(c)

        for i in range(0, len(new), 64):            # batch embedding
            batch = new[i:i + 64]
            ids = [c.metadata["chunk_id"] for c in batch]
            if vs is None:
                vs = FAISS.from_documents(batch, emb, ids=ids, normalize_L2=True)
            else:
                vs.add_documents(batch, ids=ids)
        stats["chunks"] += len(new)

        manifest[path.name] = h
        print(f"Indexed {path.name}: {len(new)} chunks")

    if vs is not None:
        index_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(index_dir))
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print("\n--- Ingestion report ---")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
