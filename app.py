from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import os
import pickle
from model import GPTConfig, GPT
from history import History, DonationHistory, FollowHistory

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def remove_caseifer(text):
    new_text = ""
    i = 0
    while i < len(text):
        if text[i] == "↨":
            if i+1 < len(text):
                new_text += text[i+1].upper()
                i += 1
        else:
            new_text += text[i]
        i += 1
    return new_text
    
def add_caseifer(text):
    tokenlist = set("\n\" !$&'#,/+=-<>*@.:;[]{}()^_?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzèé")
    replace_map = {
        "{": "[",
        "(": "[",
        "}": "]",
        ")": "]",
        "&":"and"
    }
    upperlist = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    new_text = ""
    for char in text:
        if char in tokenlist:
            if char in upperlist:
                new_text += "↨" + char.lower()
            elif char in replace_map:
                new_text += replace_map[char]
            else:
                new_text += char
    return new_text

common_line_ends = [' ', '.', ',', '?', '!', ';', ':']


model_dir = '16bit'
device = 'cuda'
dtype = 'bfloat16'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
past_history = History()
max_new_tokens = 2048 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 24 # retain only the top_k most likely tokens, clamp others to have 0 probability

ckpt_path = os.path.join(model_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

meta_path = os.path.join(model_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

class InputText(BaseModel):
    input_text: str

class GeneratedSequence(BaseModel):
    sequence: str
    
class InputWithDirection(BaseModel):
    direction: str
    input_text: str

@app.post("/generate_with_direction", response_model=GeneratedSequence)
async def generate_sequence_with_direction(input_data: InputWithDirection) -> GeneratedSequence:
    if not input_data.input_text:
        return {"sequence": ""}
    y = torch.tensor(encode(add_caseifer(input_data.direction)), dtype=torch.long, device=device)[None, ...]
    x = torch.tensor(encode(add_caseifer(input_data.input_text)), dtype=torch.long, device=device)[None, ...]
    generated_text = ""
    for idx_next in model.generate_instructed_streaming(x, y, max_new_tokens, temperature=temperature, top_k=top_k):
        char = decode([idx_next])
        if char == '\n' and generated_text in common_line_ends:
            generated_text = ""
            continue
        elif char == '\n'and not generated_text in common_line_ends:
            break
        generated_text += char
    generated_data = remove_caseifer(generated_text)
    generated_text = ''
    return {"sequence": generated_data}
    
@app.post("/history_with_direction", response_model=GeneratedSequence)
async def generate_sequence_with_direction(input_data: InputWithDirection) -> GeneratedSequence:
    if not input_data.input_text:
        return {"sequence": ""}
    past_history.add(input_data.input_text)
    y = torch.tensor(encode(add_caseifer(input_data.direction)), dtype=torch.long, device=device)[None, ...]
    x = torch.tensor(encode(add_caseifer(str(past_history)+"Emeldar: ")), dtype=torch.long, device=device)[None, ...]
    generated_text = ""
    for idx_next in model.generate_instructed_streaming(x, y, max_new_tokens, temperature=temperature, top_k=top_k):
        char = decode([idx_next])
        if char == '\n' and generated_text in common_line_ends:
            generated_text = ""
            continue
        elif char == '\n'and not generated_text in common_line_ends:
            break
        generated_text += char
    generated_data = remove_caseifer(generated_text)
    generated_text = ''
    past_history.add(generated_data)
    return {"sequence": generated_data}

@app.post("/generate", response_model=GeneratedSequence)
async def generate_sequence(input_text: InputText) -> GeneratedSequence:
    if not input_text.input_text:
        return {"sequence": ""}
    x = torch.tensor(encode(add_caseifer(input_text.input_text)), dtype=torch.long, device=device)[None, ...]
    generated_text = ""
    for idx_next in model.generate_streaming(x, max_new_tokens, temperature=temperature, top_k=top_k):
        char = decode([idx_next])
        generated_text += char
        if char == '§':
            break

    generated_data = remove_caseifer(generated_text)
    generated_text = ''
    return {"sequence": generated_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
