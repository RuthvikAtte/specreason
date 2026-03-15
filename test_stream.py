from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")
response = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    stream=True,
    stream_options={"include_usage": True}
)

for chunk in response:
    print(chunk)
