import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Retriever:
    def __init__(self, doc_path="documents"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.embeddings = []
        self.load_documents(doc_path)

    def load_documents(self, path):
        for file in os.listdir(path):
            if file.endswith(".txt"):
                with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                    text = f.read()
                    self.documents.append(text)

        self.embeddings = self.model.encode(self.documents)

    def retrieve(self, query, top_k=2):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [self.documents[i] for i in top_indices]
        return results