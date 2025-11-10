import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Get the absolute path to ../gemma relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
gemma_path = os.path.abspath(os.path.join(script_dir, "../../../dzambala/private/llm"))

# Add to PYTHONPATH
if gemma_path not in sys.path:
    sys.path.insert(0, gemma_path)

import torch
import numpy as np

import time
import argparse
import platform
import json
import shutil
import glob
import textwrap
import random
import contextlib

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

import signal
def create_graceful_exit():
    def graceful_exit(signum, frame):
        print()
        exit(0)
        # signal.signal(signum, signal.SIG_DFL)
        # os.kill(os.getpid(), signal.SIGINT)
    return graceful_exit
signal.signal(signal.SIGINT, create_graceful_exit())

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

def find_news(dataset_path, start=None, end=None):
    total_news = 0
    results = {}

    all_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    all_dirs.sort()

    # Filter by start and end if specified
    if start:
        all_dirs = [d for d in all_dirs if d >= start]
    if end:
        all_dirs = [d for d in all_dirs if d <= end]

    for folder_name in all_dirs:
        folder_path = os.path.abspath(os.path.join(dataset_path, folder_name))
        json_files = find_json_files_no_sports(folder_path)
        json_data = []
        if not json_files:
            json_data.append({'title': None, 'text_body': None})
        else:
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        data['file_path'] = json_file
                        json_data.append(data)
                    total_news = total_news + 1
                except Exception as e:
                    print(f"Failed to read {json_file}: {e}")
        results[folder_path] = json_data

    print (f'found: {total_news}')
    return results, total_news

def clear_lines(n=2):
    """Clears a specified number of lines in the terminal."""
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'
    for _ in range(n):
        sys.stdout.write(CURSOR_UP_ONE)
        sys.stdout.write(ERASE_LINE)

def format_text(text, width):
    return '\n'.join(textwrap.wrap(text, width=width))

def main():
    start_timestamp = time.time()

    parser = argparse.ArgumentParser(description='Summ training script.')

    # Required argument
    parser.add_argument('dataset_path', type=str, help='Path to the dataset')
    # Optional arguments
    parser.add_argument('--device', type=int, default=0, help='Graphics card index (default: 0)')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to the pre-trained model state dict file)')
    parser.add_argument('--size', type=int, default=48, help='Number of tokens (default: 48)')
    parser.add_argument('--cut', type=int, default=288, help='Limit input length (default: 288)')
    parser.add_argument('--start', type=str, default=None, help='Start folder)')
    parser.add_argument('--end', type=str, default=None, help='End folder)')
    parser.add_argument('--seed', type=int, default=12345, help='Random seed')
    parser.add_argument('--double', action='store_true', dest='double', default=False, help='Double encoding')
    parser.add_argument('--temperature', type=float, default=0.001, help='Weight decay (default: 0.001)')

    args = parser.parse_args()
    device = torch.device("mps") if platform.system() == 'Darwin' else torch.device(f'cuda:{args.device}')

    print ('looking for news files... ', end='', flush=True)

    results, total = find_news(args.dataset_path, args.start, args.end)

    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    from gemma import config
    from gemma import model as gemma_model

    model_config = config.get_model_config('1b')
    model_config.dtype = "float32"
    model_config.tokenizer = os.path.join(gemma_path, 'checkpoints',  'tokenizer.model')

    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if not args.ckpt:
        checkpoints = os.path.join(gemma_path, 'checkpoints', 'dzambala')
        pth_files = glob.glob(os.path.join(checkpoints, '*.pth'))
        args.ckpt = max(pth_files)
    
    print (f'using llm checkpoint {args.ckpt}')

    ts = time.time()
    with _set_default_tensor_type(model_config.get_dtype()):
        current_state_dict = {}
        print ('Loading model...', end='', flush=True)
        model = gemma_model.GemmaForCausalLM(model_config).to(device).train()
        print (f' {(time.time()-ts):.2f}s  Loading weights...', end='', flush=True)
        current_state_dict.clear()
        current_state_dict.update(
            torch.load(args.ckpt, mmap=True, weights_only=True, map_location=device)
        )
        current_state_dict['ckpt_path'] = args.ckpt
        model.load_state_dict(current_state_dict['model_state_dict'], strict=False)
        print (f' Done in {(time.time()-ts):.2f}s', flush=True)
        model.eval()
    warnings.resetwarnings()

    idx = 0
    sample = 0
    lines_to_clear = 0

    print ('\n')

    for path in sorted(results.keys()):
        
        for count, item in enumerate(results[path]):
            time_stamp = time.time()
            info = ''

            title = item.get("basic_headline")
            body = item.get("description")

            if title is None and body is None:
                continue
                '''
                hidden_list = [torch.zeros(1, 1152) for _ in range(args.size)]
                prompt = 'Empty'
                result = {'all_hidden': hidden_list, 'result': 'Empty'}
                data_time = time.time() - time_stamp
                time_stamp = time.time()
                model_time = time.time() - time_stamp
                time_stamp = time.time()
                '''
            else:
                text = f'{item["basic_headline"]}: {item["description"]}'

                prompt_enc = model.tokenizer.encode(text, bos=False)
                prompt = model.tokenizer.decode(prompt_enc[:args.cut])

                data_time = time.time() - time_stamp
                time_stamp = time.time()
                
                with torch.no_grad():
                    if args.double:
                        result = model.generate(
                            prompt, 
                            device,
                            temperature=args.temperature,   # <- greedy decoding
                            # top_k=0,
                            # top_p=1.0,
                            output_len=args.size * 2 # FLAGS.output_len
                        )
                        result = model.generate(
                            str(result["result"]), 
                            device,
                            temperature=args.temperature,   # <- greedy decoding
                            # top_k=0,
                            # top_p=1.0,
                            output_len=args.size # FLAGS.output_len
                        )

                    else:
                        result = model.generate(
                            prompt, 
                            device,
                            temperature=args.temperature,   # <- greedy decoding
                            # top_k=0,
                            # top_p=1.0,
                            output_len=args.size # FLAGS.output_len
                        )

                model_time = time.time() - time_stamp
                time_stamp = time.time()

            epoch_time = time.time() - start_timestamp
            days = int(epoch_time // (24 * 3600))
            hours = int((epoch_time % (24 * 3600)) // 3600)
            minutes = int((epoch_time % 3600) // 60)

            new = torch.stack(result['all_hidden'], dim=0).permute(1, 0, 2).squeeze(0)
            filename, _ = os.path.splitext(item['file_path'])
            news_vec_file_path = os.path.join(
                args.dataset_path,
                os.path.basename(path),
                f'{filename}.pt'
            )
            torch.save(new, news_vec_file_path)

            tail_time = time.time() - time_stamp
            clear_lines(lines_to_clear + 1)
            info += f'[{days:02}d {hours:02}:{minutes:02}], Time: {data_time:.2f}+{model_time:.2f}+{tail_time:.2f}, Timestamp: {sample+1} / {len(results.keys())} [Item: {idx+1} / {total}]\n'
            info += f'{os.path.basename(path)}: {list(new.shape)}, vector {count + 1} / {len(results[path])}'
            info += f' {os.path.basename(news_vec_file_path)}'
            info += f'\n\n'
            info += f'R: {format_text(result["result"], 100).strip()}\n\n'
            info += f'P: {format_text(prompt, 100)}\n'
            print (f'\r{info}')
            lines_to_clear = len(info.splitlines())

            idx = idx + 1
        sample = sample + 1

if __name__ == "__main__":
    main()
