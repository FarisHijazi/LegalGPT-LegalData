"""
Dataset preprocessor

this script processes lots of random unstructured files from a dataset

it does things in stages and saves intermediate results, such that if it is interrupted,
it can carry off where it left off, this also gives you the ability to inspect intermediate steps and debug

1. convert
    - split and convert all PDFs into individual images per page in their place
    - convert all xlsx and xls to CSV in their place
    - convert text files to simple JSON {text: "file content"}
2. restructure
    - load all JSON files and save them as JSON files
    - load all CSV files and convert them to JSON files
    - run OCR on all images and save the extracted text as JSON files
3. clense
    - load all JSON files and clean them up and make them more meaningful, for example putting prompts

unified structure:

represents a single file

{
    "preprocess_script_git_hash": "1234567890abcdef", // the git hash of the script that generated this file
    "schema_version": "1.0",
    "source_entity": "ministry of justice", // جهة الإصدار
    "origin_url": "http://example.com",
    "serial_number": "123456",
    "original_file_path": "path/to/file.pdf",
    "document_type": "regulation",      // نوع الوثيقة
    "circular_topic": "category",       // موضوع التعميم
    "circular_number": "123",           // رقم التعميم
    "title": "Legal Document Title",
    "issue_date": "2022-11-01",
    "effective_date": "2022-12-01",
    "expiration_date": "2023-12-31",
    "confidentiality": "public",
    "languages": ["ar", "en"],
    "contents": [
        // these are the extracted texts from the PDF
        {"text": "extracted text", "page": 1, "section": "Introduction", "text_type": "paragraph", "language": "ar"},
        {"text": "extracted text", "page": 2, "section": "Section 1", "text_type": "bullet_point", "language": "ar"}
    ],
}


"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    __doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("data_root", type=Path, help="Path to data root directory")
parser.add_argument(
    "-o",
    "--out_dir",
    default="./data/01_converted/",
    type=Path,
    help="Path to save the extracted text",
)
parser.add_argument(
    "--no_overwrite",
    default=False,
    action="store_true",
    help="Don't overwrite existing files",
)
args = parser.parse_args()

import os
import pdf2image
from tqdm.auto import tqdm
import multiprocessing
import json
import pandas as pd
from data_model import LegalDocument
from tqdm.auto import tqdm

from google.cloud import vision_v1
from concurrent.futures import ThreadPoolExecutor

import mimetypes
from PIL import Image
from PIL import ImageDraw
from io import BytesIO
from joblib import Memory
import hashlib

# Supported mime_type: application/pdf, image/tiff, image/gif
client = vision_v1.ImageAnnotatorClient()


def read_file_content(file_path):
    with open(file_path, "rb") as f:
        return f.read()


def annotate_and_save(image_path, image_response, file_content=None):
    # Get the image
    image_path = Path(image_path)
    if file_content is None:
        image = Image.open(BytesIO(file_content))
    else:
        image = Image.open(image_path)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Draw rectangles for each block
    for page in image_response.full_text_annotation.pages:
        for block in page.blocks:
            try:
                vertices = [
                    (vertex.x, vertex.y) for vertex in block.bounding_box.vertices
                ]
                draw.rectangle([vertices[0], vertices[2]], outline="red")
            except Exception as e:
                print("ERROR: Failed to draw rectangle for block", e)

    # Create the new file name
    new_file_name = (
        image_path.parent / f"{image_path.stem}.annotated{image_path.suffix}"
    )

    # Save the image
    print("saved to ", new_file_name)
    image.save(new_file_name)

    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()


# Create a memory object for caching
memory = Memory("./cachedir", verbose=0)


def hash_content(content):
    # Create a hash of the content to use as a unique identifier
    return hashlib.sha256(content).hexdigest()


@memory.cache
def google_ocr_single_image(image):
    """
    image: could be path or content
    """
    if isinstance(image, str):
        content = read_file_content(image)
        pimage = Image.open(BytesIO(content))
        mime_type, _ = mimetypes.guess_type(image)
    elif isinstance(image, bytes):
        content = image
        # guess the mime type
        pimage = Image.open(BytesIO(content))
        mime_type = "image/" + pimage.format.lower()
    else:
        raise ValueError("image should be a path or bytes")

    if mime_type is None:
        raise Exception(f"Failed to guess the mime type of {image}")

    if mime_type != "image/tiff":
        print("WARNING: Converting PNG to TIFF")
        # convert to image/tiff
        # Open the image file

        # Convert and save the image in TIFF format
        tiff_image_io = BytesIO()
        pimage.save(tiff_image_io, format="TIFF")
        tiff_image_io.seek(0)

        # Update the mime_type and content for the new TIFF image
        mime_type = "image/tiff"
        content = tiff_image_io.read()

    input_configs = [{"mime_type": mime_type, "content": content}]
    features = [{"type_": vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]

    # The service can process up to 5 pages per document file. Here we specify
    # the first, second, and last page of the document to be processed.
    requests = [
        {"input_config": input_config, "features": features, "pages": []}
        for input_config in input_configs
    ]

    results = []
    response = client.batch_annotate_files(requests=requests)
    for image_response in response.responses[0].responses:
        image_response_dict = {}
        image_response_dict["full_text"] = image_response.full_text_annotation.text
        image_response_dict["pages"] = []
        for (
            page
        ) in image_response.full_text_annotation.pages:  # this is always one page
            page_dict = {}
            page_dict["blocks"] = []
            for block in page.blocks:
                block_dict = []
                for par in block.paragraphs:
                    word_strs = " ".join(
                        [
                            "".join(symbol.text for symbol in word.symbols)
                            for word in par.words
                        ]
                    )
                    block_dict.append(word_strs)
                page_dict["blocks"].append(block_dict)
            image_response_dict["pages"].append(page_dict)
        results.append(image_response_dict)

    return results, image_response


# Update the function call to use the hashed content as the argument
google_ocr_single_image = memory.cache(google_ocr_single_image)


def process_file(file_path_inpath):
    file_path, inpath = file_path_inpath
    relative_path = os.path.relpath(file_path, inpath)
    out_path = os.path.join(args.out_dir, relative_path)

    # TODO: delete empty folders at the end
    os.makedirs(out_path, exist_ok=True)

    try:
        # TODO: maybe make stage 1 to be JSONs and then stage 2 is the OCR and the
        if file_path.suffix.lower() == ".pdf":
            images = pdf2image.convert_from_path(file_path)  # heavy work

            def process_image(i, image, out_path):
                # print('processing image', i+1, 'of', len(images))
                image_path = os.path.join(out_path, f"page_{str(i+1).zfill(3)}.tiff")
                image.save(image_path, "TIFF")

                content = read_file_content(image_path)
                results, image_response = google_ocr_single_image(content)
                try:
                    annotate_and_save(image_path, image_response, file_content=content)
                except Exception as e:
                    print("ERROR: Failed to annotate image", image_path, e)

                with open(image_path + ".annotation.json", "w") as f:
                    json.dump(results, f, indent=4)
                print("wrote to ", image_path + ".annotation.json")

            # # Create a ThreadPoolExecutor
            with ThreadPoolExecutor(
                max_workers=1
            ) as executor:  # setting this to >1 might cause joblib issues
                # Use map to execute the function in parallel
                # executor.map(process_image, zip(range(len(images)), images, [out_path]*len(images)))
                executor.map(
                    process_image, range(len(images)), images, [out_path] * len(images)
                )

        elif file_path.suffix in [".json"]:
            with open(file_path, "r") as f:
                data = f.read()
            # only write if not empty:
            if len(data) > 5:
                with open(out_path, "w") as f:
                    f.write(data)
        # TODO: if .txt don't make it .txt.txt
        elif file_path.suffix in [".txt"]:
            with open(file_path, "r") as f:
                text = f.read()
            # only write if not empty:
            if len(text) > 5:
                with open(out_path, "w") as f:
                    f.write(text)
        elif file_path.suffix in [".xls", ".xlsx", ".csv"]:
            # write as csv
            try:
                pd.read_csv(file_path).to_csv(out_path, index=False)
            except Exception as e:
                # sadly some CSV files are actually named as XLSX files
                pd.read_excel(file_path).to_csv(
                    out_path.replace(Path(out_path).suffix, ".csv"), index=False
                )
        else:
            # If the file type is not supported, print a warning message
            print(
                f"Warning: File type of {file_path.suffix} is not supported. Skipping file {file_path}."
            )
    except Exception as e:
        # If an error occurs during processing, print an error message
        print(f"Error: {e}. Skipping file {file_path}.")
        raise e


if __name__ == "__main__":
    file_paths = list(Path(args.data_root).rglob("**/*.*"))

    def init_main():
        with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.8)) as pool:
            with tqdm(total=len(file_paths)) as pbar:
                for _ in pool.imap_unordered(
                    process_file, zip(file_paths, [args.data_root] * len(file_paths))
                ):
                    pbar.update()

    init_main()
