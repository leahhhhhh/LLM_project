import openai
import gradio as gr
from openai import OpenAI
import ollama

# api_key = "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"

messages = [
    {"role": "system", "content": "You are a helpful and kind AI Assistant."},
]

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=api_key,
# )


# def chatbot(input):
#     if input:
#         messages.append({"role": "user", "content": input})
#         chat = client.chat.completions.create(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": "Say this is a test",
#                 }
#             ],
#             model="llama2",
#         )
#         reply = chat.choices[0].message.content
#         messages.append({"role": "assistant", "content": reply})

#         return reply


def ollama_(input):
    if input:
        messages.append({"role": "user", "content": input})
        response = ollama.chat(model='codellama:7b', messages=messages)
        reply=response['message']['content']
        return reply

inputs = gr.Textbox(lines=10, label="write down sth")
outputs = gr.Textbox(label="Reply")

gr.Interface(
    fn=ollama_,
    inputs=inputs,
    outputs=outputs,
    title="jw_test ollama",
    description="Ask anything you want",
    theme="compact",
).launch(share=True)


# response = ollama.chat(model='llama2', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])

