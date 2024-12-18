import logging
import random

from typing import Optional, Literal
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler

import threading
import time


class HotswapModelManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.current_model: Optional[Llama | SentenceTransformer] = None
        self.current_model_type: Optional[Literal["text", "vision", "embedding"]] = None
        self.current_model_path: Optional[str] = None
        self.current_vision_clip_model_path: Optional[str] = None
        self.current_handler: Optional[MiniCPMv26ChatHandler] = None

    def load_model(self, type: Literal["text", "vision", "embedding"], model_path: Optional[str] = None, vision_clip_model_path: Optional[str] = None):
        print("Loading model")
        logging.info("Attempting to acquire lock for model loading")
        with self.lock:
            logging.info(f"Lock acquired. Starting to load model of type: {type}")
            start_time = time.time()

            # unload current model and clip model
            if self.current_model:
                logging.info("Unloading current model")
                self.current_model.close()
                del self.current_model
                self.current_model = None
                self.current_model_type = None
                self.current_model_path = None
                self.current_vision_clip_model_path = None
            
            if self.current_handler:
                logging.info("Unloading current vision clip model")
                print(dir(self.current_handler))
                self.current_handler._llava_cpp.clip_free(self.current_handler.clip_ctx)
                del self.current_handler
                self.current_handler = None
                
            
            try:
                if type == "text":
                    logging.info(f"Loading text model from path: {model_path}")
                    model = Llama(
                        model_path=model_path, 
                        n_ctx=60000, 
                        n_gpu_layers=-1, 
                        chat_format="chatml", 
                        split_mode=2)
                elif type == "vision":
                    logging.info(f"Loading vision model from path: {model_path}")
                    logging.info(f"Using vision clip model from path: {vision_clip_model_path}")
                    handler = MiniCPMv26ChatHandler(clip_model_path=vision_clip_model_path)
                    model = Llama(
                        model_path=model_path,
                        chat_handler=handler,
                        n_ctx=20000,
                        chat_format="chatml",
                        n_gpu_layers=-1,
                        seed=random.randint(0, 1000000),
                        split_mode=2
                    )
                elif type == "embedding":
                    logging.info("Loading embedding model")
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                else:
                    raise ValueError(f"Invalid model type: {type}")

                self.current_model_type = type
                self.current_model = model
                self.current_handler = handler if type == "vision" else None
                self.current_model_path = model_path
                self.current_vision_clip_model_path = vision_clip_model_path

                end_time = time.time()
                logging.info(f"Model loaded successfully. Time taken: {end_time - start_time:.2f} seconds")
            except Exception as e:
                logging.error(f"Error loading model: {str(e)}")
                raise
        logging.info("Lock released after model loading")

    def call_model(self, type: Literal["text", "vision", "embedding"], data, model_path: Optional[str] = None, vision_clip_model_path: Optional[str] = None):
        logging.info("Attempting to acquire lock for model calling")
        with self.lock:
            logging.info(f"Lock acquired. Calling model of type: {type}")
            
            try:
                if self.current_model_type != type or self.current_model_path != model_path:
                    logging.info("Model type or path changed. Loading new model.")
                    self.load_model(type, model_path, vision_clip_model_path)

                logging.info("Executing model call")
                if type == "text":
                    return self.current_model.create_chat_completion(**data)
                elif type == "vision":
                    return self.current_model.create_chat_completion(**data)
                elif type == "embedding":
                    return self.current_model.encode(data)
                else:
                    raise ValueError(f"Invalid model type: {type}")
            except Exception as e:
                logging.error(f"Error calling model: {str(e)}")
                raise
        logging.info("Lock released after model calling")
