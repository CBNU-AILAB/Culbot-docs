import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

from utils.prompter import Prompter

import gradio as gr
import time

def respond(
        message,
        chat_history,
):
    def gen(instruction="", input_text=""):
        gc.collect()
        torch.cuda.empty_cache()
        prompt = prompter.generate_prompt_tag('1',instruction, input_text)
        output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
        s = output[0]["generated_text"]
        result = prompter.get_response(s)
        return result


    bot_message = gen(input_text=message)
    print(bot_message)
    chat_history.append((message, bot_message))
    time.sleep(0.5)
    return "", chat_history

with gr.Blocks() as demo:
    # 대충 소개글
    gr.Markdown("Cullbot")
    # 채팅 화면
    chatbot = gr.Chatbot().style(height=600)
    with gr.Row():
        with gr.Column(scale= 0.9):
            # 입력
            msg = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
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
    prompter = Prompter("cbnu2")

    demo.launch(server_name="0.0.0.0", server_port=5000)