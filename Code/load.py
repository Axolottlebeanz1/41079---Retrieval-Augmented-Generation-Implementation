import os
import nltk
nltk.download('punkt')

def load_documents(folder="data"):
    docs = []
    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            docs.append(f.read())
    return docs

def split_sentences(documents):
    sentences = []
    for doc in documents:
        sentences.extend(nltk.sent_tokenize(doc))
    return sentences
