from flask import Flask, Response
import time
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_socketio import send, emit
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
# import torch
# from threading import Thread
# import psutil
# import string
# import gc

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16
# )
# config = PeftConfig.from_pretrained("phi2")
# print(psutil.virtual_memory().percent)
# print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)
# model = AutoModelForCausalLM.from_pretrained('phi-2',quantization_config=bnb_config,
#     device_map={"": 0},
#     low_cpu_mem_usage=True,
#     trust_remote_code=True)
# print(psutil.virtual_memory().percent)
# print(psutil.virtual_memory().available * 100 / psutil.virtual_memory().total)

# model.config.use_cache = False
# model = PeftModel.from_pretrained(model, "phi2")


# tokenizer = AutoTokenizer.from_pretrained("phi-2")

# tokenizer.eos_token_id = model.config.eos_token_id
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_special_tokens({'pad_token': '<PAD>'})
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app,cors_allowed_origins='*')


def generate_words():
    # Generate words one by one
    words = ["Hello", "world", "from", "Flask", "streaming", "data"]
    for word in words:
        yield word+" "
        time.sleep(1)  # Simulate delay, replace with actual data stream logic



@socketio.on('message')
def handle_message(message):
    msgList = str(message["message"]).split(' ')
    for word in msgList:
        # yield word+" "
        emit(message['username'],word)
        time.sleep(0.2)
    emit(message['username'],"<END>")
    # send(message)


@app.route('/')
def home():
    return "Home Hellow"


@app.route('/stream_words')
def stream_words():
    return Response(generate_words(), mimetype='text/plain')

if __name__ == '__main__':
    socketio.run(app,host='0.0.0.0',port=8000,allow_unsafe_werkzeug=True)
