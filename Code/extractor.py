from transformers import pipeline

qa_pipeline = pipeline("question-answering")

def extract_answer(query, contexts):
    answers = []
    
    for context in contexts:
        try:
            result = qa_pipeline(question=query, context=context)
            answers.append((result["score"], result["answer"], context))
        except:
            continue

    if not answers:
        return None, contexts[0]

    best = max(answers, key=lambda x: x[0])
    return best[1], best[2]  # answer, source context
