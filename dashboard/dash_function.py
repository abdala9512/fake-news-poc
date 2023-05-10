
import openai

openai.api_key = "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"

def get_completion(prompt: str, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response
        