"""Using NoUse with Groq."""

import nouse
from groq import Groq

client = Groq()
brain = nouse.attach()

question = "What does this project know about Hebbian learning?"
context = brain.query(question).context_block()

response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": question},
    ],
)
print(response.choices[0].message.content)
