# download PDFs
import os
from tqdm.auto import tqdm
from multiprocessing.pool import Pool
import json
import requests


def download_file(url, outpath, position=None):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(outpath, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, leave=position is None, position=position) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))

    with open(outpath + '.done', 'w') as file:
        file.write('done')


def download_pdf(i_doc_data):
    i, (new_id, pdfUrl) = i_doc_data
    outpath = 'pdfs/' + (new_id) + '.pdf'
    if os.path.exists(outpath):
        if os.path.exists(outpath + '.done'):
            print('skipping', outpath)
            return
        else:
            print("found existing PDF file but doesn't have .done file, DELETING then redownloading", outpath)
            os.remove(outpath)

    download_file(pdfUrl, outpath, position=i + 1)
    # doc_data['pdfPath'] = outpath


if __name__ == '__main__':
    os.makedirs('pdfs', exist_ok=True)

    with open('pages_data_populated.json', 'r', encoding='utf8') as f:
        pages_data = json.load(f)

    id2pdfUrl = {x['new_id']: x['pdfUrl'] for x in pages_data['data']}

    with Pool(30) as pool:
        _ = list(tqdm(pool.imap(download_pdf, enumerate(id2pdfUrl.items())), 'downloading PDFs', len(pages_data['data']), leave=False, position=0))
