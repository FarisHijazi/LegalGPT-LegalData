import os
import pdf2image
# from pdf2image import convert_from_path
from tqdm.auto import tqdm
import multiprocessing
import argparse
from pathlib import Path
import json
import pandas as pd
import shutil


parser = argparse.ArgumentParser(description="PDF Text Extractor")
parser.add_argument("data_root", type=Path, help="Path to data root directory")
parser.add_argument("-o", "--out_dir", default="./data/01_converted/", type=Path, help="Path to save the extracted text")
args = parser.parse_args()


def process_file(file_path_inpath):
    file_path, inpath = file_path_inpath
    relative_path = os.path.relpath(file_path, inpath)
    out_path = os.path.join(args.out_dir, relative_path)

    #TODO: maybe make stage 1 to be JSONs and then stage 2 is the OCR and the 
    if file_path.suffix.lower() == ".pdf":
        os.makedirs(out_path, exist_ok=True)
        images = pdf2image.convert_from_path(file_path) # heavy work
        for i, image in enumerate(images):
            image_path = os.path.join(out_path, f"page_{str(i+1).zfill(3)}.jpg")
            image.save(image_path, "JPEG")
    # else:
    #     #TODO: deal with these later
    #     # just copy the file
    #     os.makedirs(os.path.dirname(out_path), exist_ok=True)
    #     shutil.copy(file_path, out_path)

    # elif file_path.suffix in [".json"]:
    #     with open(file_path, "r") as f:
    #         data = json.load(f)
    #     with open(out_path + ".txt", "w") as f:
    #         f.write(data.get("text", "WARNING: 'text' key not found in JSON file"))
    # elif file_path.suffix in [".txt"]:
    #     with open(file_path, "r") as f:
    #         text = f.read()
    #     with open(out_path + ".txt", "w") as f:
    #         f.write(text)
    # elif file_path.suffix in [".xls", ".xlsx", ".csv"]:
    #     # write as csv
    #     pd.read_excel(file_path).to_csv(out_path + ".csv", index=False)

    return file_path


if __name__ == '__main__':
    file_paths = list(set(
        list(Path(args.data_root).rglob("**/*.*"))
    ))

    def init_main():
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            _ = list(tqdm(pool.imap_unordered(process_file, zip(file_paths, [args.data_root]*len(file_paths))), total=len(file_paths)))

    init_main()
