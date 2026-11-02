import urllib.request
from pathlib import Path

PAPERS = {
    "attention_is_all_you_need.pdf": "1706.03762",
    "bert.pdf": "1810.04805",
    "rag_paper.pdf": "2005.11401",
    "resnet.pdf": "1512.03385",
    "adam_optimizer.pdf": "1412.6980",
}

out = Path("data/raw")
out.mkdir(parents=True, exist_ok=True)

for name, arxiv_id in PAPERS.items():
    dest = out / name
    if dest.exists():
        continue
    req = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}",
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    print("Downloaded", name)