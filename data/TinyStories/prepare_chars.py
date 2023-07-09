import os
import requests
import numpy as np
from tqdm import tqdm
import concurrent.futures
import pickle
import random
import hashlib
import shutil
from unidecode import unidecode

root_dir = r'C:\Dev\data\TinyStories'
fail_path = r'C:\Dev\data\fail'


train_file_path = os.path.join(os.path.dirname(__file__), 'train.txt')
val_file_path = os.path.join(os.path.dirname(__file__), 'val.txt')
adata = []
tdata = []
vdata = []
hash_dir = r'hashes'
baddir = r'fails'

start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"

def extract_text(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    start = -1  # Indices for start and end
    end = -1

    # Find the line numbers of start and end markers
    for i in range(len(content)):
        if start_marker in content[i]:
            start = i + 1  # You want to exclude the line with start_marker hence i + 1
        if end_marker in content[i]:
            end = i  # You want to exclude the line with end_marker hence i
        if start != -1 and end != -1:
            break  # Found both markers, no need to continue the loop
    
    if start != -1 and end != -1:
        return "".join(content[start:end])  # Join the lines between the markers
    else:
        return None
        
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

from collections import Counter

def add_caseifer(text):
    replace_map = {  # Define a mapping of characters to be replaced
    "♫": "♪"
    }
    
    # Create a list of all characters that will be used to replace unicode characters
    replace_values = ''.join(replace_map.values())
    
    # Define your set of acceptable characters (original + keys from replace_map + replace_values)
    #chars = "\n\"\t' &@!$#,/\\+=-<>*%.…_:;[]}{()^?0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz§↨©®™¶¥¼°½¾«»£βθ♪ƒ~±¤º·\x8f€¢"
    chars = "\n\t\x8f !#$%&()*+,-./:;<=>?@[\]^_{|}~§↨©®™¶¥¼°½¾«»£βθ♪ƒ±¤º·€¢\"'…0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    tokenlist = set(chars + ''.join(replace_map.keys()) + replace_values)
    #print("Tokenlist:", tokenlist)
    upperlist = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    new_text = []
    char_dist = Counter()
    excluded_chars = Counter()
    for char in text:
        if char in tokenlist:
            if char in upperlist:
                new_text.append("↨" + char.lower())
                char_dist["↨"] += 1
                char_dist[char.lower()] += 1
            elif char in replace_map:
                new_text.append(replace_map[char])
                char_dist[replace_map[char]] += 1
            else:
                new_text.append(char)
                char_dist[char] += 1
        else:
            excluded_chars[char] += 1
    return "".join(new_text), char_dist, excluded_chars




def create_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()

def is_processed(directory, file_hash):
    if os.path.isfile(os.path.join(directory, file_hash)):
        return True
    return False

def is_not_failed(directory, file_hash):
    if os.path.isfile(os.path.join(directory, file_hash)):
        return False
    return True

def is_not_processed(directory, file_hash):
    if os.path.isfile(os.path.join(directory, file_hash)):
        return False
    return True

def process_file(file_path):
    try:
        file_hash = create_hash(file_path)
        if is_not_failed(baddir, file_hash): # and is_not_processed(hash_dir, file_hash):
            try:
                with open(file_path, 'r', encoding='ansi') as f:
                    data, char_dist, excluded_chars = add_caseifer(f.read())
                    print("Character distribution:", char_dist)
                    print("Excluded characters:", excluded_chars)
            except:
                print(f"File {file_path} failed to process to format")
            try:
                filename = os.path.basename(file_path)
                folder_name = os.path.basename(os.path.dirname(file_path))
                with open("val.txt", "a", encoding='utf-8') as vfile:
                    if folder_name =="train":
                        tdata.append(data)
                        print('train' + filename)
                    elif folder_name =="test":
                        vdata.append(data)
                        print(filename)
                        vfile.write(data)
                    elif np.random.rand() < 0.995:
                        tdata.append(data)
                    else:
                        vdata.append(data)
                        vfile.write(data)
            except:
                print(f"File {file_path} failed to process to data")
            try:
                with open(os.path.join(hash_dir, file_hash), "w") as f:
                    f.write("")
            except:
                print(f"File {file_path} failed to save hash")
        else:
            shutil.move(file_path, fail_path)
    except:
        print(f"{file_path} failed to hash")

#special chars: § = section (eof) ↨ (shift key) totalling: §↨
chars = "\n\t\x8f !#$%&()*+,-./:;<=>?@[\]^_{|}~§↨©®™¶¥¼°½¾«»£βθ♪ƒ±¤º·€¢\"'…0123456789abcdefghijklmnopqrstuvwxyz"


if not os.path.exists(hash_dir):
    os.makedirs(hash_dir)
if not os.path.exists(baddir):
    os.makedirs(baddir)
if not os.path.exists(fail_path):
    os.makedirs(fail_path)
    
print("loading file list")
file_paths = []
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if file_path.endswith(".txt"):
            file_paths.append(file_path)
            
#random.shuffle(file_paths)

print("File list created beginning processeing now.")
#train_file = open(os.path.join(os.path.dirname(__file__), 'train.bin'), 'ab')
#val_file = open(os.path.join(os.path.dirname(__file__), 'val.bin'), 'ab')

for file_path in tqdm(file_paths):
    process_file(file_path)
    

print(f"Loaded files")

#valdata = add_caseifer(valdata)

#print(f"processed Validate to casified")
#valdata = ''.join([str(elem) for elem in vdata])
#traindata = ''.join([str(elem) for elem in tdata])
#traindata = add_caseifer(traindata)
#print(f"processed train to casified")
#print(f"length of dataset in characters: {len(data):,}")
#data = valdata + traindata
# get all the unique characters that occur in this text
#chars = sorted(list(set(adata)))
vocab_size = len(chars)
#print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")
print(f"Characters: {chars}")
#chars = "\n !$&',-.:;?^0123456789abcdefghijklmnopqrstuvwxyz"
adata = []

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
#n = len(data)
#train_data = data[:int(n*0.9)]
#val_data = data[int(n*0.9):]

# encode both to integers
val_ids = encode(''.join([str(elem) for elem in vdata]))
print(f"val has {len(val_ids):,} tokens")
# export to bin files
#val_ids = np.array(val_ids, dtype=np.uint16)
#val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
np.array(val_ids, dtype=np.uint16).tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
val_ids = ""

train_ids = encode(''.join([str(elem) for elem in tdata]))
print(f"train has {len(train_ids):,} tokens")
#train_ids = np.array(train_ids, dtype=np.uint16)
np.array(train_ids, dtype=np.uint16).tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)


