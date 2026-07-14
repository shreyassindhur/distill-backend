import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

def run_quick_agent(question: str, context_report: str) -> str:
    messages = [
        {
            "role": "system",
            "content": """You are a focused research assistant. The user just read a research report and has a follow-up question. 
Give a concise, direct answer in 3-5 sentences based on your knowledge. 
No report structure, no headers, no bullet points. Just a clear, direct paragraph that directly answers the question.
End with one sentence suggesting they search Distill for a full report if they want to go deeper."""
        },
        {
            "role": "user",
            "content": f"Context from their research report:\n{context_report[:1000]}\n\nFollow-up question: {question}"
        }
    ]

    for model in MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception:
            continue

    return "Unable to answer right now. Try searching this topic directly in Distill."