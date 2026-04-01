from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(query, extracted_answer, context):
    prompt = f"""
    You are a helpful assistant.

    Question:
    {query}

    Extracted Answer:
    {extracted_answer}

    Context:
    {context}

    Task:
    - Improve and clarify the answer
    - Keep it concise
    - Do NOT hallucinate beyond the context
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
