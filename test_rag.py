from src.rag_pipeline import AdvancedRAG

rag = AdvancedRAG()
rag.ingest("data/Abdullah.yasser.pdf")

questions = [
    "What is the person's name?",
    "What university did he graduate from?",
    "What programming languages does he know?",
]

for q in questions:
    result = rag.ask(q)
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print()