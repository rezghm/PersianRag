import os
import json
from openai import OpenAI
from jdatetime import datetime

datetime_now = datetime.now().strftime("%Y%m%d")
LOG_FILE = f"logs/llm_calls{datetime_now}.log"


def log_call(model_name, prompt, output):
    entry = {
        "timestamp": datetime.now().strftime("%H%M%S"),
        "model": model_name,
        "input": prompt,
        "output": output,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def call_llm(model_name, api_key, base_url, prompt):
    if not api_key:
        raise RuntimeError("API_KEY not set")
    
    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    output = resp.choices[0].message.content
    log_call(model_name, prompt, output)
    return output
