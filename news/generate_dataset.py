import os
import sys
import time
import yfinance as yf

from pprint import pprint

import trafilatura
import uuid
import json
import random

from collections import deque

from openai import OpenAI

client = OpenAI(
    api_key = "mysecretkey"
)

full_path  = '/mnt/StorageMedia/dzambala_summ/part_012'
if not os.path.isdir(full_path):
    os.makedirs(full_path)

from llm.gemma import tokenizer
tokenizer = tokenizer.Tokenizer(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'llm',
        'checkpoints',
        'tokenizer.model'
    )
)

def ask_gpt(prompt: str, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def uid_short(text: str, length: int = 24) -> str:
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:length]

def find_json_files_no_sports(root_dir):
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if '.sports.' in filename:
                continue
            if filename.endswith('.json'):
                json_files.append(
                    os.path.abspath(
                        os.path.join(dirpath, filename))
                    )
    return json_files

def get_folder_range(start_path, end_path=None):
    """
    Get list of folders from start_path to end_path (inclusive).
    If end_path is None, returns just [start_path].
    """

    from pathlib import Path

    start_path = Path(start_path)
    
    # Get parent directory
    parent_dir = start_path.parent
    
    # If end_path provided, check it has same parent
    if end_path is not None:
        end_path = Path(end_path)
        if end_path.parent != parent_dir:
            raise ValueError(f"Paths must have same parent folder: {parent_dir} != {end_path.parent}")
    
    # Get all folders in parent directory (sorted)
    all_folders = sorted([
        f for f in parent_dir.iterdir() 
        if f.is_dir()
    ])
    
    # Get folder names
    folder_names = [f.name for f in all_folders]
    
    # Find indices
    start_name = start_path.name
    if start_name not in folder_names:
        raise ValueError(f"Folder '{start_name}' not found in {parent_dir}")
    
    start_idx = folder_names.index(start_name)
    
    if end_path is None:
        return [all_folders[start_idx]]
    
    end_name = end_path.name
    if end_name not in folder_names:
        raise ValueError(f"Folder '{end_name}' not found in {parent_dir}")
    
    end_idx = folder_names.index(end_name)
    
    # Return range (inclusive)
    if start_idx <= end_idx:
        return all_folders[start_idx:end_idx + 1]
    else:
        return all_folders[end_idx:start_idx + 1]

gpt_prompt = '''
            Summarise in set of separate terms and keywords, 
            avoid reporting numbers and dates but keep facts and codes, 
            begin with mention finantial area or area of knowledge, 
            country and continent, 
            report overall feeling in terms of financial sentiment, 
            do not use "**", always include short summary
            Preserve consistent formatting for key: value with keys "Financial area", "Country", "Continent", "Financial sentiment", "Summary" Do not report number of tokens
            '''

VMIN = 1
VMAX = 2
TMIN = 12
TMAX = 14
MAXPROMPT = 144

folder1 = sys.argv[1]
folder2 = sys.argv[2] if len(sys.argv) > 2 else None

try:
    result = get_folder_range(folder1, folder2)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

for folder_name in result:
    json_files = find_json_files_no_sports(os.path.abspath(folder_name))

    for json_file in sorted(json_files):

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        f.close()

        # print (data)
        # sys.exit()

        item = {}
        summ = ask_gpt(f'{gpt_prompt}, return in {random.randint(TMIN, TMAX)} tokens: {data["description"]}')
        item['prompt'] = data['basic_headline'] + '. ' + data["description"]
        item['summ'] = summ.strip()

        filename = 'reuters.' + os.path.basename(json_file)

        with open(os.path.join(full_path, filename), "w") as f:
            json.dump(item, f, indent=4, default=str)
        print ('=====')
        print (filename)
        print (item.get('title'))
        print (item.get('summ'))
        print ('----------')
        print (item.get('prompt'))

        time.sleep(1)
