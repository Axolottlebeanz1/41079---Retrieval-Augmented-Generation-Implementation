from ingest import load_documents, split_sentences
from retriever import build_bm25, retrieve
from extractor import extract_answer
from generator import generate_answer

# Step 1: Load data
docs = load_documents()
sentences = split_sentences(docs)

# Step 2: Build BM25 index
bm25, _ = build_bm25(sentences)

# Step 3: Query loop
while True:
    query = input("\nAsk a question (or 'exit'): ")
    if query.lower() == "exit":
        break

    # Step 4: Retrieve
    contexts = retrieve(query, bm25, sentences)

    # Step 5: Extract
    extracted_answer, best_context = extract_answer(query, contexts)

    # Step 6: Generate (optional refinement)
    final_answer = generate_answer(query, extracted_answer, best_context)

    print("\nFinal Answer:\n", final_answer)
