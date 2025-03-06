from openai import OpenAI

client = OpenAI(
  api_key="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"
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message)
