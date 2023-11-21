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

    format_path = "format.txt"

    with open(format_path, "r", encoding="utf-8") as file:
        format = file.read()

    example_path = "example.txt"

    with open(example_path, "r", encoding="utf-8") as file:
        example = file.read()   

    info__dir_path = "/Users/kim-yeongsang/Desktop/instructino_ai/information/"
    for i in range(0,66):
        print("현재: {}".format(i))
        info_path = info__dir_path + "information_{}.txt".format(i)

        with open(info_path, "r", encoding="utf-8") as file:
            information = file.read()

        system_prompt = """
        You are an AI that generates data based on information. 
        """
        prompt = f"""
        Create data of a specific structure based on the information I provide. 

        The information is as follows: 
        `
            {information}
        `

        The data structure is in the following JSON format. 
        `   
            {format}
        `

        Here are some examples of the data:
        `
            {example}
        `

        You need to comply with the following requirements.
        requirements:
        1. The output should be an appropriate response to the instruction and the input. Make sure the output is less than 100 words.
        2. The content of the generated data should not be duplicated.
        3. All data (instruction, input, output) should be written in Korean.
        4. Create 10 pieces of data and arrange them in a list format.
        5. Please provide the answer without interruption and within the limited token range.
        6. Not all instructions require input. For example, when a instruction asks about some general information, "what is the highest peak in the world", it is not necssary to provide a specific context. In this case, we simply put "" in the input field.
        """
        try:
            response = openai_call(system_prompt, prompt, max_tokens=2000)
            # print(response)
            data = json.loads(response)
        except:
            print("범위를 넘었습니다")
            continue

        file_name_format = "data_{}.json"

        file_number = 1
        while True:
            try:
                file_name = file_name_format.format(file_number)
                file_path = os.path.join("./data", file_name)
                if not os.path.exists(file_path):
                    with open(file_path, "w", encoding="utf-8") as json_file:
                        json.dump(data, json_file, ensure_ascii=False, indent=4)
                        print(f"The result save to {file_path}")
                    break
                file_number += 1        
            except:
                print("failed save file {}".format(file_number))

        # with open("data.json", "a", encoding="utf-8") as json_file:
        #     json.dump(data, json_file, ensure_ascii=False, indent=4)
        #     json_file.write("\n")

if __name__ == "__main__":
    main()