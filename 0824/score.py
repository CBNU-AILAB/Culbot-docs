# -*- coding: utf-8 -*-

from dotenv import load_dotenv
import tiktoken as tiktoken
import openai
import os
import time
import json


# Load default environment variables (.env)
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")).lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))


openai.api_key = OPENAI_API_KEY


def limit_tokens_from_string(string: str, model: str, limit: int) -> str:
    """Limits the string to a number of tokens (estimated)."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.encoding_for_model('gpt2')  # Fallback for others.

    encoded = encoding.encode(string)

    return encoding.decode(encoded[:limit])


def openai_call(
    system_prompt: str,
    prompt: str,
    model: str = LLM_MODEL,
    temperature: float = OPENAI_TEMPERATURE,
    max_tokens: int = 100,
):
    while True:
        try:
            if not model.lower().startswith("gpt-"):
                # Use completion API
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].text.strip()
            else:
                # Use 4000 instead of the real limit (4097) to give a bit of wiggle room for the encoding of roles.
                # TODO: different limits for different models.

                trimmed_prompt = limit_tokens_from_string(prompt, model, 4000 - max_tokens)

                # Use chat completion API
                messages = [{"role": "system" , "content": system_prompt},{"role": "system", "content": trimmed_prompt}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(
                "   *** The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.Timeout:
            print(
                "   *** OpenAI API timeout occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIError:
            print(
                "   *** OpenAI API error occurred. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.APIConnectionError:
            print(
                "   *** OpenAI API connection error occurred. Check your network settings, proxy configuration, SSL certificates, or firewall rules. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.InvalidRequestError:
            print(
                "   *** OpenAI API invalid request. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        except openai.error.ServiceUnavailableError:
            print(
                "   *** OpenAI API service unavailable. Waiting 10 seconds and trying again. ***"
            )
            time.sleep(10)  # Wait 10 seconds and try again
        else:
            break




def main():

    data_test_path = "/Users/kim-yeongsang/Desktop/instructino_ai/data_test_pred/data_test_transform_pred_gpt_rag.json"
    with open(data_test_path, 'r', encoding="utf-8") as file:
        data_test = json.load(file)
    
    score_list = []
    
    for i, d in enumerate(data_test):
        print("현재: {}".format(i))
      
        system_prompt = """
        you are an AI that measures scores for predicted answers to questions as a medical expert.
        """
        prompt = f"""

        "Based on the medical facts you possess and the given question-answer set, please measure the 'output predicted by the AI model' accuracy scores  from a medical perspective on a scale of 0 to 10."

        Question: 
        `
            {d['instruction']}
        `

        Answer:
        `   
            {d['output']}
        `

        The output predicted by the AI model:
        `
            {d['pred']}
        `

        The output response only contains score.
        """
        try:
            response = openai_call(system_prompt, prompt, max_tokens=2000)
            score_list.append({"id": i, "instruction" : d['instruction'], "output": d['output'], 'pred':d['pred'], "score": response })
            print(response)
        except:
            print("올바르지 않은 값")
            continue
        time.sleep(1)
    
    with open('/Users/kim-yeongsang/Desktop/instructino_ai/data_test_score/score_transform_gpt_rag.json', "w", encoding="utf-8") as json_file:
        json.dump(score_list, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()