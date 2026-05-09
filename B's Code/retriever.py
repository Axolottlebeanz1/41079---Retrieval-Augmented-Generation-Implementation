import os
import fitz
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class Retriever:
    def __init__(
        self,
        doc_path="documents",
        allowed_extensions=None
    ):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.doc_sources = []
        self.embeddings = []

        if allowed_extensions is None:
            allowed_extensions = [".txt", ".docx", ".pdf"]

        self.allowed_extensions = allowed_extensions

        self.load_documents(doc_path)

    def load_documents(self, path):
        for file in os.listdir(path):
            full_path = os.path.join(path, file)

            # Skip files user is not allowed to access
            if not any(
                file.endswith(ext)
                for ext in self.allowed_extensions
            ):
                continue

            # -------------------------
            # TXT FILES
            # -------------------------
            if file.endswith(".txt"):
                with open(
                    full_path,
                    "r",
                    encoding="utf-8"
                ) as f:
                    text = f.read()

                    sentences = text.split(". ")
                    for sentence in sentences:
                        if len(sentence.strip()) > 20:
                            self.documents.append(sentence.strip())
                            self.doc_sources.append(file)

            # -------------------------
            # DOCX FILES
            # -------------------------
            elif file.endswith(".docx"):
                doc = Document(full_path)
                text = "\n".join(
                    [para.text for para in doc.paragraphs]
                )

                sentences = text.split(". ")
                for sentence in sentences:
                    if len(sentence.strip()) > 20:
                        self.documents.append(sentence.strip())
                        self.doc_sources.append(file)

            # -------------------------
            # PDF FILES
            # -------------------------
            elif file.endswith(".pdf"):
                pdf = fitz.open(full_path)
                text = ""

                for page in pdf:
                    text += page.get_text()

                pdf.close()

                sentences = text.split(". ")
                for sentence in sentences:
                    if len(sentence.strip()) > 20:
                        self.documents.append(sentence.strip())
                        self.doc_sources.append(file)

        if self.documents:
            self.embeddings = self.model.encode(self.documents)

            self.doc_to_id = {
                doc: f"doc_{i}"
                for i, doc in enumerate(self.documents)
            }

    def retrieve(self, query, top_k=3):
        if not self.documents:
            return []

        query_embedding = self.model.encode([query])

        similarities = cosine_similarity(
            query_embedding,
            self.embeddings
        )[0]

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []

        for i in top_indices:
            chunk = self.documents[i]

            snippet = (
                chunk[:120] + "..."
                if len(chunk) > 120
                else chunk
            )

            results.append({
                "full": chunk,
                "snippet": snippet,
                "score": similarities[i],
                "source": self.doc_sources[i]
            })

        return results