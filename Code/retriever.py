from rank_bm25 import BM25Okapi
import nltk

def tokenize(text):
    return nltk.word_tokenize(text.lower())

def build_bm25(sentences):
    tokenized = [tokenize(s) for s in sentences]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized

def retrieve(query, bm25, sentences, top_k=5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [sentences[i] for i in ranked[:top_k]]
