from pdf2image import convert_from_path
import os
import base64
import requests
import os
import json
from tqdm.auto import tqdm
from multiprocessing.pool import ThreadPool
import time
import glob
import argparse
import json
import pandas as pd
import numpy as np


def page_range(page_spec):
    """Parse a string containing pages and ranges into a list of integers."""
    page_list = []
    # Split the input on commas to process multiple parts like "1,2,3" or "1-3,5"
    parts = page_spec.split(',')
    for part in parts:
        if '-' in part:
            # If part is a range like "1-10"
            start, end = part.split('-')
            page_list.extend(range(int(start), int(end) + 1))
        else:
            # If part is a single page number
            page_list.append(int(part))
    return page_list


parser = argparse.ArgumentParser()
parser.add_argument('document', type=str, help='pdf or image path')
parser.add_argument('--output_dir', type=str, default='{document}.output', help='Output directory to store images')
parser.add_argument('--cols', nargs='+', type=str, default=[], help='List of column names')
parser.add_argument('--parallelism', type=int, default=10, help='parallelism')
parser.add_argument('--pages', type=page_range, default=[], help='List of pages or page ranges (e.g., 1-3,5,7)')
args = parser.parse_args()

for col in args.cols:
    assert ',' not in col, 'cannot have comma "," in --cols, use spaces to separate columns'

args.output_dir = args.output_dir.format(
    document=args.document,
)

# OpenAI API Key
api_key = os.environ['OPENAI_API_KEY']
# api_key = 'sk-'

# Create output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

if args.document.endswith('.pdf'):
    print('Extracting PDF pages')
    # Convert PDF to a list of images
    images = convert_from_path(args.document)

    # Save each page as an image
    for i, image in enumerate(images):
        image_path = os.path.join(args.output_dir, f'page_{i+1}.png')
        image.save(image_path, 'PNG')

    print(f'Images are extracted and saved in {args.output_dir}.')

    impaths = sorted(glob.glob(f'{args.output_dir}/*.png'))
else:
    impaths = [args.document]

# Function to encode the image
def encode_image(impath):
    with open(impath, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def markdown_table_to_df(table_str):
    # Split the string into lines
    lines = table_str.strip().split('\n')
    lines = [l.strip().strip('|') for l in lines]

    # Extract the header using the first line
    headers = [header.strip() for header in lines[0].split('|') if header.strip()]

    # Extract the data rows
    rows = [[value.strip() for value in line.split('|') if value.strip()] for line in lines[2:]]  # Skip the second line as it is the separator

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df


def go(impath):

    if args.pages:
        if not any([f'page_{p}.png' in impath for p in args.pages]):
            print('skipping excluded page:', impath)
            return

    # Getting the base64 string
    base64_image = encode_image(impath)

    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}

    payload = {
        'model': 'gpt-4-turbo',
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': f"""Please transcribe this image clearly as a concise markdown table (no padding spaces). Make sure you go through every single entry in the table, skip nothing.
Note that there are exactly {len(args.cols)} columns: {args.cols}
""",
                    },
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}},
                ],
            }
        ],
        'max_tokens': 4096,
    }

    def proc(response_json):
        table_string = response_json['choices'][0]['message']['content']
        print(table_string)

        splits = table_string.split('```')
        if len(splits) > 1:
            table_string = splits[1].strip().lstrip('markdown').strip()
        else:
            table_string = ('|' + '|'.join(table_string.split('|')[1:-1]) + '|').strip()

        with open(impath + '.json', 'w', encoding='utf8') as f:
            json.dump(response_json, f)

        df = markdown_table_to_df(table_string)
        assert df.shape[1] == len(args.cols), f'expected {len(args.cols)} columns but instead got {df.shape[1]} with {impath}'
        df.to_csv(impath + '.csv', index=False)
        return df

    if os.path.exists(impath + '.json'):
        try:
            print('trying to see if result already exists', impath + '.json')
            with open(impath + '.json', 'r', encoding='utf8') as f:
                response_json = json.load(f)
            return proc(response_json)
        except Exception:
            print('unable to properly load existing JSON')
            # os.remove(impath+".json")

    retries = 3
    for retry in range(retries):
        try:
            # print("running", impath)
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload)
            response_json = response.json()
            df = proc(response_json)
            print('done')
            return df
        except Exception as e:
            print(f'Error:', e, impath)
            print(response_json)
            # print('llm_answer:', run.run_data.llm_answer)
            print(f'Retrying {retry+1}/{retries}...')
            time.sleep(5)
    # else:
    #     raise Exception(f"Tried {retries} times and still failed")


dfs = list(tqdm(ThreadPool(args.parallelism).imap(go, impaths), total=len(impaths)))
dfs = [d for d in dfs if d is not None]

combined = pd.DataFrame(
    data=np.vstack([d.values for d in dfs]),
    columns=args.cols,
)

combined.to_csv(f'{args.output_dir}/combined.csv', index=False)
print('saved to', f'{args.output_dir}/combined.csv')
