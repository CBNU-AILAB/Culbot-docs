import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel, PeftConfig
import fire

from utils.prompter import Prompter

from dotenv import load_dotenv
import gradio as gr
import time
import pandas as pd
import openai
import chromadb
import tiktoken as tiktoken
import os

LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")).lower()
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai.api_key = OPENAI_API_KEY


# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.

# Note. alternatively you can set a temporary env variable like this:
# os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")


##
import json
from pathlib import Path
from pprint import pprint

import pandas as pd
df = pd.read_json('')
df = df.astype({'vector_id':'str'})
import chromadb

chroma_client = chromadb.Client()

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Test that your OpenAI API key is correctly set as an environment variable
# Note. if you run this notebook locally, you will need to reload your terminal and the notebook for the env variables to be live.

# Note. alternatively you can set a temporary env variable like this:
# os.environ["OPENAI_API_KEY"] = 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")


embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name='text-embedding-ada-002')

question_collection = chroma_client.create_collection(name='question', embedding_function=embedding_function)

batch_size = 166
chunks = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]


for chunk in chunks:
    question_collection.add(
        ids=chunk['vector_id'].tolist(),
        embeddings=chunk['question_vector'].tolist(),  # Assuming you have the 'question_vector' column
    )

def query_collection(collection, query, max_results, dataframe):
    results = collection.query(query_texts=query, n_results=max_results, include=['distances']) 
    df = pd.DataFrame({
                'id':results['ids'][0], 
                'score':results['distances'][0],
                'fileName': dataframe[dataframe.vector_id.isin(results['ids'][0])]['fileName'],
                'content': dataframe[dataframe.vector_id.isin(results['ids'][0])]['content'],
                'question': dataframe[dataframe.vector_id.isin(results['ids'][0])]['question'],
                })
    return df


##

def respond(
        message,
        chat_history,
):
    def gen(instruction="", input_text=""):
        gc.collect()
        torch.cuda.empty_cache()
        prompt = prompter.generate_prompt(instruction, input_text)
        output = pipe(prompt, max_length=1024, temperature=0.2, num_beams=5, eos_token_id=2)
        s = output[0]["generated_text"]
        result = prompter.get_response(s)
        return result
    
    output = query_collection(
        collection=question_collection,
        query=message,
        max_results=1,
        dataframe=df
        )

    context = ''
    for c in output.content:
        context += c
    print(context)
    bot_message = gen(instruction=message, input_text=context)
    chat_history.append((message, bot_message))
    time.sleep(0.5)
    return "", chat_history

with gr.Blocks() as demo:
    # 대충 소개글
    gr.Markdown("Culbot")
    # 채팅 화면
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale= 0.9):
            # 입력
            msg = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            )
        with gr.Column(scale=0.1):
            # 버튼
            clear = gr.Button("➤")
    # 버튼 클릭
    clear.click(respond, [msg, chatbot], [msg, chatbot])
    # 엔터키
    msg.submit(respond, [msg, chatbot], [msg,chatbot])

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    MODEL = "EleutherAI/polyglot-ko-12.8b"

    LORA_WEIGHTS = ""

    model = AutoModelForCausalLM.from_pretrained(MODEL, load_in_8bit=True,device_map={"":0})
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS)
    model.eval()

    pipe = pipeline("text-generation", model=model, tokenizer=MODEL)
    prompter = Prompter("cbnu")

    demo.launch(server_name="0.0.0.0", server_port=8081)
