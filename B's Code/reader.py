from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Reader:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    def generate_answer(self, query, retrieved_chunks):
        if not retrieved_chunks:
            return "Sorry, I couldn't find relevant information."

        # Handle dict or string
        if isinstance(retrieved_chunks[0], dict):
            chunks = [c["full"] for c in retrieved_chunks[:2]]
        else:
            chunks = retrieved_chunks[:2]

        context = "\n".join(chunks)

        prompt = f"""
Answer the question clearly and concisely using the context.

Context:
{context}

Question: {query}

Answer:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer.strip()