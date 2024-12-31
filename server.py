from flask import jsonify, request, Flask
import flask_cors
from llama_cpp.llama import Llama, LlamaGrammar
from threading import Thread
from PIL import Image
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer
# from libs.extractor import extract_text_from_image
import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
from libs.common import generate_grammar
from pathlib import Path
import random
import os
import time
import json
import threading


import logging
from typing import Optional, Literal

# pip install -U pymilvus
from pymilvus import MilvusClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from libs.hotswap_model_manager import HotswapModelManager

from pathlib import Path

def get_first_available_model(base_path: str, model_names: list[str]) -> str:
    for name in model_names:
        path = Path(base_path) / name
        if path.is_file():
            return str(path)
    raise FileNotFoundError(f"No available model found in {base_path}")

text_models = [
    "Rombos-LLM-V2.6-Qwen-14b-Q8_0.gguf",
    "Rombos-LLM-V2.6-Qwen-14b-Q6_K.gguf",
    "Rombos-LLM-V2.5-Qwen-42b-Q4_K_M.gguf",
    "Rombos-LLM-V2.6-Qwen-14b-Q2_K_L.gguf",
    "Qwen2.5-14B-Instruct-Q6_K_L.gguf",
    "Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q8_0.gguf",
    "Llama-3.2-3B-Instruct-Q8_0.gguf",
    "Llama-3.2-3B-Instruct-Q6_K_L.gguf",
    "medicine-Llama3-8B.Q8_0.gguf"
]

vision_models = [
    "minicpm/ggml-model-Q8_0.gguf",
    "minicpm/ggml-model-Q6_K.gguf"
]

model_path = get_first_available_model("../models/text", text_models)
vision_model_path = get_first_available_model("../models/vision", vision_models)

vision_clip_model_path = "../models/vision/minicpm/mmproj-model-f16.gguf"

def image_to_base64_data_uri(base64_image):
    return f"data:image/png;base64,{base64_image}"

def init_llm():
    #return Llama(model_path=model_path, n_ctx=20000, n_gpu_layers=-1, chat_format="chatml", split_mode=2)
    pass

#pip install sentence-transformers
def init_embedding_llm():
    # Load the all-MiniLM-L6-v2 model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    """
    modelRoot = "../models/embedding/"
    # create folder if not exists
    os.makedirs(modelRoot, exist_ok=True)
    model_path = modelRoot + "stella_en_1.5B_v5"

    if os.path.exists(model_path):
        model = SentenceTransformer(model_path).cuda()
    else:

        model = SentenceTransformer('dunzhang/stella_en_1.5B_v5')

        model.save(model_path)
        model = SentenceTransformer(model_path).cuda()
    """
    return model


def create_app():
    app = Flask(__name__)
    flask_cors.CORS(app)
    chat_llm = init_llm()
    embedding_llm = init_embedding_llm()
    hotswap_manager = HotswapModelManager()
    @app.route("/vision", methods=['POST'])
    def vision_endpoint():
        global vision_model_path, vision_clip_model_path
        data = request.get_json()
        image = data['image']
        img_bytes = base64.b64decode(image)
        img = Image.open(BytesIO(img_bytes))

        user_prompt = data.get('user_prompt', None)
        system_prompt = data.get('system_prompt', None)
        chat_history = data.get('chat_history', None)


        img_bytes = base64.b64decode(image)
        img = Image.open(BytesIO(img_bytes))

        # scale image to 1806336 total pixels keeping aspect ratio
        total_final_pixels = 1806336
        width, height = img.size
        aspect_ratio = width / height
        new_height = int((total_final_pixels / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)
        img = img.resize((new_width, new_height))

        # convert scaled image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        image_data_uri = image_to_base64_data_uri(img_str)

        messages = [
            {"role": "system", "content": system_prompt}
        ]
        if chat_history:
            messages.extend(chat_history)

        messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri }},
                    {"type": "text", "text": user_prompt}
                ]
            })

        response = hotswap_manager.call_model(
            type="vision",
            data={
            "messages": messages,
            "max_tokens": 1000
            },
            model_path=vision_model_path,
            vision_clip_model_path=vision_clip_model_path
        )
        return jsonify(response)

    @app.route("/vision_batch", methods=['POST'])
    def vision_batch_endpoint():
        global vision_model_path, vision_clip_model_path
        data = request.get_json()
        images = data['images']
        user_prompt = data.get('user_prompt', None)
        system_prompt = data.get('system_prompt', None)
        chat_history = data.get('chat_history', None)
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        if chat_history:
            messages.extend(chat_history)

        final_message = {
            "role": "user",
            "content": []
        }

        for image in images:
            img_bytes = base64.b64decode(image)
            img = Image.open(BytesIO(img_bytes))

            # scale image to 1806336 total pixels keeping aspect ratio
            total_final_pixels = 1806336
            width, height = img.size
            aspect_ratio = width / height
            new_height = int((total_final_pixels / aspect_ratio) ** 0.5)
            new_width = int(new_height * aspect_ratio)
            img = img.resize((new_width, new_height))

            # convert scaled image to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            image_data_uri = image_to_base64_data_uri(img_str)

            final_message['content'].append(
                    {"type": "image_url", "image_url": {"url": image_data_uri }}
            )
            break

        final_message['content'].append( {"type": "text", "text": user_prompt} )
        messages.append(final_message)

        print("~" * 100)
        print(messages)
        print("~" * 100)

        response = hotswap_manager.call_model(
            type="vision",
            data={
            "messages": messages,
            "max_tokens": 1000
            },
            model_path=vision_model_path,
            vision_clip_model_path=vision_clip_model_path
        )
        return jsonify(response)

    @app.route('/chat', methods=['POST'])
    def chat_endpoint():
        data = request.get_json()
        messages = data['messages']
        grammar_text = data.get('grammar')
        temperature = data.get('temperature')

        grammar = None
        if grammar_text and len(grammar_text) > 0:
            print("grammar received")

            grammar = LlamaGrammar.from_string(grammar_text, verbose=True)

        try:
            if temperature is not None:
                result = chat_llm.create_chat_completion(
                    messages=messages,
                    grammar=grammar,
                    temperature=temperature
                )
            else:
                result = chat_llm.create_chat_completion(
                    messages=messages,
                    grammar=grammar
                )
        except Exception as e:
            print("~" * 100)
            print(e)
            print("~" * 100)
            return jsonify({'error': str(e), 'chat': None})

        return jsonify({'error': None, 'chat': result})

    # background_chat endpoint, just like /chat but uses hotswap model loading
    @app.route('/background_chat', methods=['POST'])
    def background_chat_endpoint():
        global model_path
        data = request.get_json()
        messages = data['messages']
        grammar_text = data.get('grammar')
        temperature = data.get('temperature')
        grammar_schema = data.get('grammar_schema')

        # get model path from data if it exists
        model_path = data.get('model_path', model_path)
        grammar = None
        if grammar_text and len(grammar_text) > 0:
            grammar = LlamaGrammar.from_string(grammar_text, verbose=True)

        if grammar_schema:
            grammar = LlamaGrammar.from_schema(grammar_schema, verbose=True)

        try:
            if temperature is not None:
                result = hotswap_manager.call_model(
                    type="text",
                    data={
                        "messages": messages,
                        "grammar": grammar,
                        "temperature": temperature
                    },
                    model_path=model_path
                )
            else:
                result = hotswap_manager.call_model(
                    type="text",
                    data={
                        "messages": messages,
                        "grammar": grammar
                    },
                    model_path=model_path
                )
        except Exception as e:
            print("~" * 100)
            print(e)
            print("~" * 100)
            return jsonify({'error': str(e), 'chat': None})

        print("~" * 100)
        print(result)
        print("~" * 100)
        return jsonify({'error': None, 'chat': result})

    # embeds text using the SentenceTransformer model
    @app.route('/embed', methods=['POST'])
    def embed_endpoint():
        data = request.get_json()
        text = data['text']
        embedding = embedding_llm.encode(text)
        return jsonify(embedding.tolist())

    # embeds batch of texts using the SentenceTransformer model
    @app.route('/embed_batch', methods=['POST'])
    def embed_batch_endpoint():
        data = request.get_json()
        texts = data['texts']
        embeddings = embedding_llm.encode(texts)
        return jsonify(embeddings.tolist())

    @app.route("/transcribe_with_vision", methods=['POST'])
    def transcribe_endpoint():
        global vision_model_path, vision_clip_model_path
        data = request.get_json()
        image = data['image']
        img_bytes = base64.b64decode(image)
        img = Image.open(BytesIO(img_bytes))

        # scale image to 1806336 total pixels keeping aspect ratio
        total_final_pixels = 1806336
        width, height = img.size
        aspect_ratio = width / height
        new_height = int((total_final_pixels / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)
        img = img.resize((new_width, new_height))

        # convert scaled image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        image_data_uri = image_to_base64_data_uri(img_str)

        query = data.get('query')
        query_template = ""

        query_params = [
            {"name": "markdown", "type": "string"}
        ]
        query_grammar = generate_grammar(query_params)

        vision_system_prompt = r"""You are a Vision Language Model that accurately transcribes image text to markdown. Adhere to these rules:

1. Transcribe all visible text exactly, preserving spelling and punctuation.
2. Use markdown formatting (#headings, **bold**, *italic*, lists, >quotes, ```code```).
3. Follow logical reading order (usually top-to-bottom, left-to-right).
4. Preserve layout within markdown constraints.
5. Describe diagrams/charts briefly in [brackets].
6. Mark unclear text as [unclear] or [partially obscured].
7. Exclude irrelevant text (e.g., watermarks) unless requested.
8. Use LaTeX for equations: $equation$.
9. Note [handwritten] for significant handwritten text.

Respond with only the markdown-formatted transcription. Do not include any additional text or explanations."""

        vision_user_prompt = r"""Transcribe the text in the provided image to markdown format."""

        messages = [
           {"role": "system", "content": vision_system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri }},
                    {"type": "text", "text": vision_user_prompt}
                ]
            }
        ]

        response = hotswap_manager.call_model(
            type="vision",
            data={
            "messages": messages,
            "max_tokens": 1000
            },
            model_path=vision_model_path,
            vision_clip_model_path=vision_clip_model_path
        )

        return jsonify(response)

    @app.route('/', methods=['GET'])
    def info_endpoint():
        model = Path(model_path).stem
        vision_model = Path(vision_model_path).stem
        vision_clip_model = Path(vision_clip_model_path).stem
        return f'''
        <html>
            <body>
                <h2>LLM Server Info</h2>
                <p>Chat model: <strong>{model}</strong></p>
                <p>Vision model: <strong>{vision_model}</strong></p>
                <p>Vision clip model: <strong>{vision_clip_model}</strong></p>
            </body>
        </html>
        '''

    return app

def run_app(port):
    app = create_app()
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == '__main__':
    threads = []
    for i in range(1):
        t = Thread(target=run_app, args=(5000 + i,))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()
