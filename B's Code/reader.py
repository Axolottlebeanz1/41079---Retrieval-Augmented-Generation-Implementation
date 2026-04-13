class Reader:
    def generate_answer(self, query, docs):
        context = "\n".join(docs)

        answer = f"""
Question: {query}

Relevant Information:
{context}

Answer:
Based on the documents, {context}
"""
        return answer