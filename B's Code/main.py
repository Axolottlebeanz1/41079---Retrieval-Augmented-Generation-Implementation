from retriever import Retriever
from reader import Reader

def main():
    retriever = Retriever()
    reader = Reader()

    while True:
        query = input("\nWhat is your question?: ")

        if query.lower() == "exit":
            break

        docs = retriever.retrieve(query)
        answer = reader.generate_answer(query, docs)

        print("\n--- Relevant Documents ---")
        for d in docs:
            print("-", d)

        print("\n--- Answer ---")
        print(answer)

if __name__ == "__main__":
    main()