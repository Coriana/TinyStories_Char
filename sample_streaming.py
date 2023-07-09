"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import time
import random
import pyttsx3
import concurrent.futures
import sys
from time import sleep
import socket
import re
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('samples.db')
c = conn.cursor()

# Create a table to store the samples
c.execute('''CREATE TABLE IF NOT EXISTS samples
             (history text, chosen_output text, rejected_sample text)''')
MAX_HISTORY_LENGTH = 2048

class History:
    def __init__(self):
        self.lines = []
        self.length = 0
        self.name = ""
        self.direction = ""
    
    def add(self, line):
    # If the line is '§\n', clear the history
        if line == '§\n':
            self.lines = []
            self.length = 0
            self.name = ""
            
        line_length = len(line)
        while (self.length + line_length) > (MAX_HISTORY_LENGTH - len(self.direction)) and self.length >=2:
            self.length -= len(self.lines.pop(0))
        self.lines.append(line)
        self.length += line_length
        os.system(f"title {self.length}/{MAX_HISTORY_LENGTH}")
    
    def __str__(self):
        history = self.direction + "\n" + "".join(self.lines)
        if len(history) > MAX_HISTORY_LENGTH:
            history = history[-MAX_HISTORY_LENGTH:]
        return history
        
def send_data(data, port):
    s = socket.socket()
    s.connect(('localhost', port))
    s.send(data.encode())
    s.close()
    
def remove_caseifer(text):
    new_text = ""
    i = 0
    while i < len(text):
        if text[i] == "↨":
            if i+1 < len(text):
                new_text += text[i+1].upper()
                i += 1
            else:
                pass  # skip this index
        else:
            new_text += text[i]
        i += 1
    return new_text
    
def add_caseifer(text):
   
    # Define your set of acceptable characters (original + keys from replace_map + replace_values)
    #chars = "\n\"\t' &@!$#,/\\+=-<>*%.…_:;[]}{()^?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz§↨©®™¶¥¼°½¾«»£βθ♪ƒ~±¤º·\x8f€¢"
    tokenlist = "\n\t\x8f !#$%&()*+,-./:;<=>?@[\]^_{|}~§↨©®™¶¥¼°½¾«»£βθ♪ƒ±¤º·€¢\"'…0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    upperlist = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    new_text = []
    for char in text:
        if char in tokenlist:
            if char in upperlist:
                new_text.append("↨" + char.lower())
            else:
                new_text.append(char)
        else:
            pass
    return "".join(new_text)

    
#def Read_input():
    # do this

#def ContinueOutput():
    # do that

#def StartFromNothing():
    # do the other thing
# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'TinyStories2048_768' # ignored if init_from is not 'resume'
start = "§\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 4096 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1336
MAX_LENGTH = 1024
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
direction_file = "direction.txt"
follower_file = "follower.txt"
input_file = "input.txt"
autorun_file = "autoplay.txt"
sample_file = "sampler.txt"

#torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
past_history = History()

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join(out_dir, 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
text = ''
history = start
#past_history.direction = "I am Emeldar and today is a nice day.\n"
past_history.add(start)

# run generation
with torch.no_grad():
    with ctx:
        while True:
            with open(f"log_{out_dir}.txt", "a") as f:
                text = input(">:")
                if text != '':
                    start_ids = encode(add_caseifer(text))
                else:
                    start_ids = encode(add_caseifer(start))
                #history = str(past_history) # append input to history
                #start_ids = encode(add_caseifer(start))
               # print(history)
                x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
                generated_text = []
                shift = False
                for idx_next in model.generate_streaming(x, max_new_tokens, temperature=temperature, top_k=top_k):
                    # convert the index to a character and print it to the screen
                    char = decode([idx_next])
                    # append the character to the generated text
                    generated_text.append(char)

                    # check for newline character
                    if char == '§':
                        generated_text.append('\n')
                        # append the completed line to the list or print it to the screen
                   #     generated_sequences.append(generated_text)
                        # reset the generated text for the next line
                        
                        #generated_text = []
                        break
                actual_output = remove_caseifer("".join(generated_text))
                print(actual_output, end='', flush=True)
                f.write(actual_output)
                #past_history.add(actual_output)
                text = ''


