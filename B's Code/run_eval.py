from retriever import Retriever
from eval import load_eval_dataset, run_eval

retriever = Retriever()

dataset = load_eval_dataset(
    "RAG_Test_Questions_Answers.txt"
)

run_eval(
    retriever,
    dataset,
    top_k=2
)