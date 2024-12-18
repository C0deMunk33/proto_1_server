import requests
import base64
import json


def generate_format_description(params):
    description = "{ \n"
    for param in params:
        description += f"\"{param['name']}\" : {param['type']}\n"
    description += "}"
    return description

""" Example:
params = [
    {"name": "name", "type": "string"},
    {"name": "age", "type": "string"},
    {"name": "hobbies", "type": "array"}
]

# should generate
# {
#   "name" : "string",
#   "age" : "string",
#   "hobbies" : "array"
# }
"""
def generate_grammar(params):
    grammar = r"""root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  ("{" ws ("""
    
    add_comma = False
    for param in params:
        if add_comma:
            grammar += r""""," ws """
        else:
            add_comma = True
        grammar += rf""" "\"{param['name']}\" : " {param['type']} """

    grammar += r""") "}"  )

bool  ::= ("true" | "false") ws
    
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws

string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
  )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

int ::= "-"? [0-9]+ ws

str ::= string

# whitespace
ws ::= ([ \t\n] ws)?"""
    return grammar

def create_grammar_enum(enum_class):
    # should take a class with class variables as enum values
    # return "(\"value1\" | \"value2\" | \"value3\")"
    enum = "("
    add_pipe = False
    for key, value in enum_class.__dict__.items():
        if not key.startswith("__"):
            if add_pipe:
                enum += " | "
            else:
                add_pipe = True
            enum += f"\"{value}\""
    enum += ")"
    return enum
    
def call_llm(llm_url, system_prompt, messages, grammar=None, temperature=None):
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            'messages': [{"role": "system", "content": system_prompt}] + messages,
            'max_tokens': 30000,
            'stream': False
        }
        
        if grammar is not None and grammar != "":
            data['grammar'] = grammar
        
        if temperature is not None and temperature != "":
            data['temperature'] = temperature
        
        final_url = llm_url + "/chat"

        response = requests.post(final_url, headers=headers, json=data)
        response_data = response.json()

        if 'chat' in response_data:
            result_text = response_data['chat']['choices'][0]['message']['content']
            return result_text
    
    except Exception as error:
        print("~~~~~~~~~~~~~~~~~~~~~~~")
        print("Error")
        print(error)
        print("~~~~~~~~~~~~~~~~~~~~~~~")
        return "error"
    
def call_background_chat(server_url, type, messages, grammar=None, model_path=None, temperature=None):
    try:
        headers = {
            'Content-Type': 'application/json'
        }
        data = {
            'messages': messages,
            'temperature': temperature
        }
        
        if model_path is not None and model_path != "":
            data['model_path'] = model_path

        if grammar is not None and grammar != "":
            data['grammar'] = grammar
        
        final_url = server_url + f"/background_chat"

        response = requests.post(final_url, headers=headers, json=data)
        response_data = response.json()

        if 'chat' in response_data:
            result_text = response_data['chat']['choices'][0]['message']['content']
            return result_text
    
    except Exception as error:
        print("~~~~~~~~~~~~~~~~~~~~~~~")
        print("Error")
        print(error)
        print("~~~~~~~~~~~~~~~~~~~~~~~")
        return "error"

def file_to_url_encoded_base64(file_path):
    #load file bytes
    with open(file_path, "rb") as image_file:
        image_bytes = image_file.read()
    #encode to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    #encode to url encoded base64
    url_encoded_base64 = base64.urlsafe_b64encode(base64_image).decode('utf-8')
    return url_encoded_base64

def hosted_image_to_url_encoded_base64(hosted_image_url):
    response = requests.get(hosted_image_url)
    #encode to base64
    base64_image = base64.b64encode(response.content).decode('utf-8')
    #encode to url encoded base64
    url_encoded_base64 = base64.urlsafe_b64encode(base64_image).decode('utf-8')
    return url_encoded_base64

def embed(server_url, text):
    embed_url = server_url + "/embed"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'text': text
    }
    response = requests.post(embed_url, headers=headers, json=data)
    return response.json()

def embed_batch(server_url, texts):
    embed_url = server_url + "/embed_batch"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'texts': texts
    }
    response = requests.post(embed_url, headers=headers, json=data)
    return response.json()

def vision(server_url, user_prompt, system_prompt, base64_image, chat_history=None):
    vision_url = server_url + "/vision"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'user_prompt': user_prompt,
        'system_prompt': system_prompt,
        'image': base64_image
    }
    if chat_history is not None:
        data['chat_history'] = chat_history
    response = requests.post(vision_url, headers=headers, json=data)
    return response.json()[0]["content"]["choices"][0]["message"]["content"]

def vision_batch(server_url, images, user_prompt, system_prompt, chat_history=None):
    vision_url = server_url + "/vision_batch"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'images': images,
        'user_prompt': user_prompt,
        'system_prompt': system_prompt
    }
    if chat_history is not None:
        data['chat_history'] = chat_history
    response = requests.post(vision_url, headers=headers, json=data)
    

    return response.json()["choices"][0]["message"]["content"]


def ocr(server_url, base64_image):
    ocr_url = server_url + "/transcribe_with_vision"
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        'image': base64_image
    }
    response = requests.post(ocr_url, headers=headers, json=data)
    return response.json()


def summarize_and_extract_key_points(llm_server, chat_history):
    summary_params = [
        {"name": "paragraph_summary", "type": "string"},
        {"name": "sentence_summary", "type": "string"},
        {"name": "key_points", "type": "array"},
        {"name": "key_words", "type": "array"}
    ]
    summary_format = generate_format_description(summary_params)
    summary_grammar = generate_grammar(summary_params)
    system_prompt = rf"""You are a knowledge worker who has been asked to summarize the following chat transcript into a single paragraph and a single sentence. Additionally, you are to extract key points and key words from the chat.

Reply in JSON in the following format:
{summary_format}

Ensure you use well formatted JSON with field names and values in double quotes.
"""
    user_prompt = "Please, summarize the current chat history up to this point into a single paragraph and a single sentence and extract key points as an array of sentences and key words as an array of words."
    messages = chat_history + [{ "role": "user", "content": user_prompt}]
    #call_background_chat(server_url, type, messages,grammar=None, model_path=None, temperature=None):
    response = call_background_chat(llm_server, "text", [{"role": "system", "content": system_prompt}] + messages, summary_grammar)
    print("~" * 80)
    print(response)
    print("~" * 80)
    response_json = json.loads(response)
    return response_json

def split_into_sentences(text):
    # TODO: make better
    return text.split('.')


def external(func):
    return func


















if __name__ == "__main__":
    print("Common functions for LLM server")
    #embed
    print(embed("http://localhost:5001", "hello this is a test"))
