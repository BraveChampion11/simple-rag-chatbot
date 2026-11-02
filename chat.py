from src.rag import RAG

rag = RAG()
print("RAG chatbot ready. Type 'exit' to quit.\n")
while True:
    q = input("You: ").strip()
    if q.lower() in {"exit", "quit"}:
        break
    if not q:
        continue
    print("Bot: ", end="", flush=True)
    sources = []
    for token, sources in rag.ask(q):
        print(token, end="", flush=True)
    print()
    if sources:
        print("Sources:", ", ".join(f"{s} (p.{p})" for s, p in sources))
    print()
