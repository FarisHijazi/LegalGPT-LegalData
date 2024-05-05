# %%
import argparse
import datetime
import hashlib
import json
import os
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path

import requests
from tqdm.auto import tqdm

cookies = {
    'XSRF-TOKEN': 'eyJpdiI6IjZoWTJPQk5sRXdPVUprL3pFNDVXdHc9PSIsInZhbHVlIjoiNHJMNU5kcXpPVGNDUWN1aHdFTitZanU3d3hpeWxub1dOdnoyZHZ0UmQ2eVhpOS9nVTErWnVvMEJJY3lScGI5d3k4RVZmY05KTzkzSkxBNTJpVjFseWhiQmh3UW9oeE5LQnJ6U0J3U2NLUjhZWkREQ1BNRXRVNG8za0h0VkIzZ2kiLCJtYWMiOiJiMDE5MTYwOGEzOGQ0MzAyNTcxMjQ4MjQzOGJhMDExNzQ4MDlmNWY2MTYxOWJjYWVjNDMyOTNmZTg4ODdjZWU4IiwidGFnIjoiIn0%3D',
    'laravel_session': 'eyJpdiI6Illmdzk1MWFYVmtKN3kzY2FVaDVxU1E9PSIsInZhbHVlIjoiWEhlS0E3VnRINHluaUZkTC93MVR0WnQ0S3Fyd2NHa1hCcTBvaVBHMG9PaXJiczFSdktObXhMWFkxZHl2VkltaDlCQVphcVRiM2cxeGhaR3UwUXkrdkZFdE5sVVJQUk5oUEMwMzhvcHUvR2NUNk5keXpYcWhPL1JCSCtYM2lsaS8iLCJtYWMiOiIzZDQ5YWI0YTZlZmE0MjVmMWYyMmYzOGVmMmYzZjFkYTE2YmY3YTM1NTk3MWZhZmI1YTk0YjkzYTZjNjlmMWU1IiwidGFnIjoiIn0%3D',
}

headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    # 'cookie': 'XSRF-TOKEN=eyJpdiI6IjZoWTJPQk5sRXdPVUprL3pFNDVXdHc9PSIsInZhbHVlIjoiNHJMNU5kcXpPVGNDUWN1aHdFTitZanU3d3hpeWxub1dOdnoyZHZ0UmQ2eVhpOS9nVTErWnVvMEJJY3lScGI5d3k4RVZmY05KTzkzSkxBNTJpVjFseWhiQmh3UW9oeE5LQnJ6U0J3U2NLUjhZWkREQ1BNRXRVNG8za0h0VkIzZ2kiLCJtYWMiOiJiMDE5MTYwOGEzOGQ0MzAyNTcxMjQ4MjQzOGJhMDExNzQ4MDlmNWY2MTYxOWJjYWVjNDMyOTNmZTg4ODdjZWU4IiwidGFnIjoiIn0%3D; laravel_session=eyJpdiI6Illmdzk1MWFYVmtKN3kzY2FVaDVxU1E9PSIsInZhbHVlIjoiWEhlS0E3VnRINHluaUZkTC93MVR0WnQ0S3Fyd2NHa1hCcTBvaVBHMG9PaXJiczFSdktObXhMWFkxZHl2VkltaDlCQVphcVRiM2cxeGhaR3UwUXkrdkZFdE5sVVJQUk5oUEMwMzhvcHUvR2NUNk5keXpYcWhPL1JCSCtYM2lsaS8iLCJtYWMiOiIzZDQ5YWI0YTZlZmE0MjVmMWYyMmYzOGVmMmYzZjFkYTE2YmY3YTM1NTk3MWZhZmI1YTk0YjkzYTZjNjlmMWU1IiwidGFnIjoiIn0%3D',
    'dnt': '1',
    'if-modified-since': 'Mon, 26 Feb 2024 10:48:43 GMT',
    'if-none-match': '"8037965da168da1:0"',
    'sec-ch-ua': '"Chromium";v="123", "Not:A-Brand";v="8"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    # 'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'authorization': 'bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MTQ1NTU3MTIsInN1YiI6IjkzZjE5MTJiLTQzNzEtNGEyYS05NTg2LWUzM2RjMzYxMDNlZCJ9.NjV4YYREnf8RawkyEaDuneBTnvjmoovlBTl6bhomgbo',
}


def hash_str(input_string: str):
    return hashlib.sha256(input_string.encode('utf-8')).hexdigest()


def json_remove_keys(json_obj, keys):
    if isinstance(json_obj, dict):
        return {k: json_remove_keys(v, keys) for k, v in json_obj.items() if k not in keys}
    if isinstance(json_obj, list):
        return [json_remove_keys(v, keys) for v in json_obj]
    return json_obj


def hash_json(json_obj):
    # Serialize JSON object with sorted keys
    serialized_json = json.dumps(json_obj, sort_keys=True).encode('utf-8')
    return hashlib.sha256(serialized_json).hexdigest()


def get_id(item):
    return hash_json(
        {
            'title_en': item['title_en'],
            'title_ar': item['title_ar'],
            'number': item['number'],
            'approve_date': item['Approves'][0]['approve_date'],
        }
    )


def populate_page_data(page_data):
    """
    populate with pageUrl, pdfUrl, and fetches documentData
    """
    page_data['pageUrl'] = 'https://ncar.gov.sa/document-details/' + page_data['id']
    page_data['pdfUrl'] = 'https://ncar.gov.sa/api/index.php/resource/' + page_data['id'] + '/Documents/OriginalAttachPath'

    if 'documentData' in page_data and page_data['documentData'] is not None:
        return

    retries = 5

    for retry in range(retries):
        try:
            response = requests.get('https://ncar.gov.sa/api/index.php/api/documents/document/' + page_data['id'], headers=headers)
            assert response.status_code == 200, response.text
            page_data['documentData'] = response.json()
            return
        except Exception as e:
            print(f"Failed to get page data for {page_data['id']} on retry {retry} due to {e}")
            time.sleep(5)
            continue
    else:
        print(f"Failed to get page data for {page_data['id']} after {retries} retries")
        return


# %%


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape ncar.gov.sa')
    parser.add_argument(
        '--use_cached', action='store_true', help='Will use locally existing pages_data.json and pages_data_populated.json if they already exist'
    )
    parser.add_argument('--output_dir', type=Path, default='.', help='Output directory to save the scraped data, default is the current directory')
    args = parser.parse_args()

    print('fetching all_release_orgnizations ...')
    response = requests.post('https://ncar.gov.sa/api/index.php/api/documents/all_release_orgnizations/1/1', headers=headers, cookies=cookies)
    partial_data = response.json()
    # print('partial_data', len(partial_data), partial_data)
    response = requests.post(
        'https://ncar.gov.sa/api/index.php/api/documents/all_release_orgnizations/1/' + str(partial_data['dataLength']),
        headers=headers,
        cookies=cookies,
    )
    all_release_orgnizations = response.json()
    with open(args.output_dir / 'all_release_orgnizations.json', 'w', encoding='utf8') as f:
        json.dump(all_release_orgnizations, f, ensure_ascii=False, indent=4)

    print('saved all_release_orgnizations.json', len(all_release_orgnizations['data']))

    # %%

    if not os.path.exists(args.output_dir / 'ar_i18n.json') and not args.use_cached:
        response = requests.get('https://ncar.gov.sa/assets/i18n/ar.json', headers=headers)
        with open(args.output_dir / 'ar_i18n.json', 'w') as f:
            f.write(response.text)
        print('saved ar_i18n.json')

    if os.path.exists(args.output_dir / 'pages_data.json') and args.use_cached:
        print('--use_cached: loading existing pages_data.json ...')
        with open(args.output_dir / 'pages_data.json', 'r', encoding='utf8') as f:
            pages_data = json.load(f)
    else:
        print('fetching data ...')
        response = requests.get('https://ncar.gov.sa/api/index.php/api/documents/list/1/1/approveDate/DESC', headers=headers, cookies=cookies)
        partial_data = response.json()

        response = requests.get(
            'https://ncar.gov.sa/api/index.php/api/documents/list/1/' + str(partial_data['dataLength']) + '/approveDate/DESC',
            headers=headers,
            cookies=cookies,
        )
        pages_data = response.json()
        print('pulled data:', len(pages_data['data']))

        with open(args.output_dir / 'pages_data.json', 'w', encoding='utf8') as f:
            json.dump(pages_data, f, ensure_ascii=False)
        print('saved pages_data.json', len(pages_data['data']), 'entries')

    # %%

    # the "new_id" is the id of the document, and the "id" is a randomly generated string for the URLs
    global_new_id2pagedata = {}
    # global_stupid_id2new_id = {}

    if os.path.exists(args.output_dir / 'pages_data_populated.json') and args.use_cached:
        with open(args.output_dir / 'pages_data_populated.json', 'r', encoding='utf8') as f:
            pages_data = json.load(f)
            print('--use_cached: loaded old populated data')
    else:
        print('populating pages data ...')

        with ThreadPool(5) as pool:
            _ = list(tqdm(pool.imap(populate_page_data, pages_data['data']), 'populating pages data', len(pages_data['data'])))

        failures = []
        for page_data in pages_data['data']:
            new_id = get_id(page_data)
            if new_id in global_new_id2pagedata:
                print(f'DUPLICATE NEW ID: {new_id}')

            page_data['new_id'] = new_id
            page_data['pdfPath'] = 'pdfs/' + (page_data['new_id']) + '.pdf'
            global_new_id2pagedata[page_data['new_id']] = page_data
            # global_stupid_id2new_id[page_data['id']] = new_id

            for documentRelation in page_data['documentData']['data']['documentRelation']:
                try:
                    new_id = get_id(documentRelation)
                    if new_id in global_new_id2pagedata:
                        print(f'DUPLICATE NEW ID: {new_id}')

                    documentRelation['new_id'] = new_id
                    global_new_id2pagedata[documentRelation['new_id']] = documentRelation
                    # global_stupid_id2new_id[documentRelation['id']] = new_id
                except Exception as e:
                    failures.append((page_data, documentRelation, e))
                    print(f'Failed to process {documentRelation} due to {e}')

        if failures:
            print('WARNING: there were failures in processing documentRelation', len(failures))

    pages_data['meta'] = {
        'scrape_date': datetime.datetime.now().isoformat(),
    }
    with open(args.output_dir / 'pages_data_populated.json', 'w', encoding='utf8') as f:
        json.dump(pages_data, f, ensure_ascii=False, indent=4)

    id2pdfUrl = {x['new_id']: x['pdfUrl'] for x in pages_data['data']}
    print('total number of documents', len(pages_data['data']))
    print('unique new_ids', len(id2pdfUrl))
    percentage_of_conflicts = (len(pages_data['data']) - len(id2pdfUrl)) / len(pages_data['data']) * 100
    print(f'percentage of conflicts ID {percentage_of_conflicts:.2f}%')
