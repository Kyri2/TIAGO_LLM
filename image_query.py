
import os
import sys
from openai import OpenAI
import base64
import requests
api_key = "your_api_key"
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def prompt_condition(file):
    with open(file, 'r', encoding='utf-8') as file:
        prompt = file.read()
    return prompt
# Path to your image
def gpt_image(image_path,pmt):
    base64_image = encode_image(image_path)
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    payload = {
      "model": "chatgpt-4o-latest",
      "temperature": 0.0,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"{pmt}"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 3000
    }


    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    answer = response.json()["choices"][0]["message"]["content"]
    return answer

def check_action_affordance(image_path,action,environment_description):
    base64_image = encode_image(image_path)
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    file_path = os.path.join(script_dir, 'affordance_check_prompt')
    pmt = prompt_condition(file_path)
    pmt += prompt_condition('robot_action_lib')
    pmt += environment_description
    question = f"Can the action '{action}' be performed on the object in the image?just answer 'yes' or 'no'. for example: if you provided with move_hand(chocolate bar), you should answer 'yes' if the action can be performed by the robot arm and 'no' if it can't be performed. please provide explanation in a short sentence if answer is no"
    pmt += question
    payload = {
      "model": "chatgpt-4o-latest",
      "temperature": 0.0,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f"{pmt}"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(response.json())
    answer = response.json()["choices"][0]["message"]["content"]
    return answer

if __name__ == '__main__':
    env = gpt_image('oranges.jpg',' ')
    print(env)
    # print(check_action_affordance('case.jpg','move_hand(chocolate bar)',env))