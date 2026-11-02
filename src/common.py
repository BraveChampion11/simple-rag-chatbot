from pathlib import Path
import yaml
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_embeddings(cfg):
    return OllamaEmbeddings(model=cfg["embed_model"])


def load_store(cfg):
    """Return the saved FAISS index, or None if nothing is indexed yet."""
    p = Path(cfg["paths"]["index"])
    if (p / "index.faiss").exists():
        return FAISS.load_local(str(p), get_embeddings(cfg),
                                allow_dangerous_deserialization=True,
                                normalize_L2=True)
    return None