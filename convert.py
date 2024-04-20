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

"""

import argparse
import io
import json
import mimetypes
import multiprocessing
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pandas as pd
import pdf2image
from google.cloud.vision_v1.types.image_annotator import AnnotateImageResponse
from joblib import Memory
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_root', type=Path, help='Path to data root directory')
parser.add_argument(
    '-o',
    '--out_dir',
    default='./data/01_converted/',
    type=Path,
    help='Path to save the extracted text',
)
parser.add_argument(
    '--no_overwrite',
    default=False,
    action='store_true',
    help="Don't overwrite existing files",
)
parser.add_argument(
    '--no_ocr_cache',
    default=False,
    action='store_true',
    help="Don't use OCR cache, will run OCR on all images even those it's seen before",
)
# add argument list to whitelist specific file extensions
parser.add_argument(
    '--ocr',
    default='google',
    choices=['google', 'azure'],
    help='choose OCR provider: google or azure',
)
# add argument list to whitelist specific file extensions
parser.add_argument(
    '--whitelist',
    default=['.pdf', '.xls', '.xlsx', '.csv', '.txt', '.json'],
    nargs='+',
    help='List of file extensions to whitelist',
)
# set number of concurrent threads/processes
parser.add_argument(
    '-p',
    '--processors',
    default=int(multiprocessing.cpu_count() * 0.8),
    type=int,
    help='Number of concurrent processes to run',
)
args = parser.parse_args()


if args.ocr == 'google':
    from google.cloud import vision_v1

    # Supported mime_type: application/pdf, image/tiff, image/gif
    google_vision_client = vision_v1.ImageAnnotatorClient()
else:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials

    computervision_client = ComputerVisionClient(os.environ['VISION_ENDPOINT'], CognitiveServicesCredentials(os.environ['VISION_KEY']))


def read_file_content(file_path):
    with open(file_path, 'rb') as f:
        return f.read()


def get_git_hash():
    """get git hash of the current commit for this file convert.py"""
    # check if there are any changes to the current file, if so, raise an exception
    if subprocess.check_output(['git', 'status', '--porcelain', __file__]):
        raise Exception(
            'ERROR: There are uncommitted changes to this file, please commit them before running this script, this is to verify script and data integrity.'
        )
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')


# get git hash:
GIT_HASH = get_git_hash()
# GIT_HASH = None


def google_ocr_annotate_and_save(image_path, image_response, file_content=None):
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
                vertices = [(vertex.x, vertex.y) for vertex in block.bounding_box.vertices]
                draw.rectangle([vertices[0], vertices[2]], outline='red')
            except Exception as e:
                print('\nERROR: Failed to draw rectangle for block', e)

    # Create the new file name
    new_file_name = image_path.parent / f'{image_path.stem}.annotated{image_path.suffix}'

    # Save the image
    print('saved to ', new_file_name)
    image.save(new_file_name)


def azure_ocr_annotate_and_save(image_path, read_result, file_content=None):
    # Get the image
    image_path = Path(image_path)
    if file_content is None:
        image = Image.open(BytesIO(file_content))
    else:
        image = Image.open(image_path)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Loop over each line in the result
    for text_result in read_result.analyze_result.read_results:
        for line in text_result.lines:
            # Get the bounding box coordinates
            bounding_box = line.bounding_box

            # Draw rectangle
            draw.rectangle([(bounding_box[0], bounding_box[1]), (bounding_box[4], bounding_box[5])], outline='red')

    # Create the new file name
    new_file_name = image_path.parent / f'{image_path.stem}.annotated{image_path.suffix}'

    # Save the image
    image.save(new_file_name)
    print('azure_ocr_annotate_and_save() saved to ', new_file_name)


# Create a memory object for caching
memory = Memory('./cachedir', verbose=0)


@memory.cache
def google_ocr_single_image(image: bytes) -> AnnotateImageResponse:
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
        mime_type = 'image/' + pimage.format.lower()
    else:
        raise ValueError('image should be a path or bytes')

    if mime_type is None:
        raise Exception(f'Failed to guess the mime type of {image}')

    if mime_type != 'image/tiff':
        print('WARNING: Converting PNG to TIFF')
        # convert to image/tiff
        # Open the image file

        # Convert and save the image in TIFF format
        tiff_image_io = BytesIO()
        pimage.save(tiff_image_io, format='TIFF')
        tiff_image_io.seek(0)

        # Update the mime_type and content for the new TIFF image
        mime_type = 'image/tiff'
        content = tiff_image_io.read()

    input_configs = [{'mime_type': mime_type, 'content': content}]
    features = [{'type_': vision_v1.Feature.Type.DOCUMENT_TEXT_DETECTION}]

    # The service can process up to 5 pages per document file. Here we specify
    # the first, second, and last page of the document to be processed.
    requests = [{'input_config': input_config, 'features': features, 'pages': []} for input_config in input_configs]

    response = google_vision_client.batch_annotate_files(requests=requests)
    assert len(response.responses[0].responses) == 1, 'FATAL ERROR: something is wrong with the number of responses'
    image_response = response.responses[0].responses[0]

    return image_response


if args.no_ocr_cache:
    google_ocr_single_image = memory.cache(google_ocr_single_image)


@memory.cache
def azure_ocr_single_image(image: bytes):
    '''
    OCR: Read File using the Read API, extract text - remote
    This example will extract text in an image, then print results, line by line.
    This API call can also extract handwriting style text (not shown).
    '''
    read_response = computervision_client.read_in_stream(io.BytesIO(image), raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers['Operation-Location']
    # Grab the ID from the URL
    operation_id = read_operation_location.split('/')[-1]

    # Call the "GET" API and wait for it to retrieve the results
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Print the detected text, line by line
    # if read_result.status == OperationStatusCodes.succeeded:
    #     for text_result in read_result.analyze_result.read_results:
    #         for line in text_result.lines:
    #             print(line.text)
    #             print(line.bounding_box)

    return read_result


if args.no_ocr_cache:
    azure_ocr_single_image = memory.cache(azure_ocr_single_image)


def process_file(file_path_inpath):
    file_path, inpath = file_path_inpath
    relative_path = os.path.relpath(file_path, inpath)
    out_path = os.path.join(args.out_dir, relative_path)

    # TODO: delete empty folders at the end
    os.makedirs(out_path, exist_ok=True)

    if file_path.suffix not in args.whitelist:
        # print(f'Warning: File is not whitelisted. Skipping file. "{file_path}"')
        return

    try:
        # TODO: maybe make stage 1 to be JSONs and then stage 2 is the OCR and the
        if file_path.suffix.lower() == '.pdf':
            images: list[Image.Image] = pdf2image.convert_from_path(file_path)  # heavy work

            def process_image(i, image, out_path):
                # print('processing image', i+1, 'of', len(images))
                image_path = os.path.join(out_path, f'page_{str(i+1).zfill(3)}.tiff')
                if args.no_overwrite and os.path.exists(image_path):
                    print('--no_overwrite, skipping existing file:', image_path)
                    return

                # print('saving to', image_path)
                image.save(image_path, 'TIFF')

                def PIL_to_bytes(image, format='TIFF'):
                    # a hack
                    byte_arr = io.BytesIO()
                    image.save(byte_arr, format=format)
                    return byte_arr.getvalue()

                content = PIL_to_bytes(image, format='TIFF')

                if args.ocr == 'google':
                    image_response = google_ocr_single_image(content)
                    try:
                        google_ocr_annotate_and_save(image_path, image_response, file_content=content)
                    except Exception as e:
                        print('\nERROR: Failed to annotate image', image_path, e)
                    contents = [
                        {
                            'text': image_response.full_text_annotation.text,
                            'page': i + 1,
                            'section': None,  # TODO: infer this value
                            'text_type': None,  # TODO: infer this value
                            'languages': [x.language_code for x in image_response.full_text_annotation.pages[i].property.detected_languages],
                        }
                        for i in range(len(image_response.full_text_annotation.pages))  # don't worry there's only one page
                    ]
                else:
                    read_result = azure_ocr_single_image(content)
                    try:
                        azure_ocr_annotate_and_save(image_path, read_result, file_content=content)
                    except Exception as e:
                        print('\nERROR: Failed to annotate image', image_path, e)
                    # TODO: double check this
                    contents = [
                        {
                            'text': line.text,
                            'page': i + 1,
                            'section': None,
                            'text_type': None,
                            'languages': None,
                        }
                        for i, text_result in enumerate(read_result.analyze_result.read_results)
                        for line in text_result.lines
                    ]

                with open(image_path + '.organized.json', 'w') as f:
                    organized_object = {
                        'preprocess_script_git_hash': GIT_HASH,
                        'schema_version': '1.0',
                        'source_entity': None,  # TODO: infer this value
                        'origin_url': None,  # TODO: infer this value
                        'serial_number': None,  # TODO: infer this value
                        'original_file_path': str(file_path),
                        'document_type': None,  # TODO: infer this value
                        'circular_topics': None,  # TODO: infer this value
                        'circular_number': None,  # TODO: infer this value
                        'title': None,  # TODO: infer this value
                        'issue_date': None,  # TODO: infer this value
                        'effective_date': None,  # TODO: infer this value
                        'expiration_date': None,  # TODO: infer this value
                        'confidentiality': None,  # TODO: infer this value
                        'languages': None,  # TODO: infer this value
                        'contents': contents,
                    }
                    json.dump(organized_object, f, indent=4)
                    print('wrote to ', image_path + '.organized.json')

            # # Create a ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=1) as executor:  # setting this to >1 might cause joblib issues
                # Use map to execute the function in parallel
                # executor.map(process_image, zip(range(len(images)), images, [out_path]*len(images)))
                executor.map(process_image, range(len(images)), images, [out_path] * len(images))
        elif file_path.suffix.lower() in ['.json']:
            if args.no_overwrite and os.path.exists(out_path):
                print('--no_overwrite, skipping existing file:', out_path)
                return
            with open(file_path, 'r') as f:
                data = f.read()
            # only write if not empty:
            if len(data) > 5:
                json_object = json.loads(data)
                if type(json_object) is not dict and not type(json_object[0]) is dict:
                    print(f'Warning: File is not a valid JSON file. Skipping file. "{file_path}"')
                    return

                def get_content_from_json(json_object):
                    if 'full_text' in json_object:
                        return json_object['full_text']
                    elif 'text' in json_object:
                        return json_object['text']
                    else:
                        return None

                with open(out_path + '.organized.json', 'w') as f:
                    organized_object = {
                        'preprocess_script_git_hash': GIT_HASH,
                        'schema_version': '1.0',
                        'source_entity': None,  # TODO: infer this value
                        'origin_url': None,  # TODO: infer this value
                        'serial_number': None,  # TODO: infer this value
                        'original_file_path': str(file_path),
                        'document_type': None,  # TODO: infer this value
                        'circular_topics': None,  # TODO: infer this value
                        'circular_number': None,  # TODO: infer this value
                        'title': None,  # TODO: infer this value
                        'issue_date': None,  # TODO: infer this value
                        'effective_date': None,  # TODO: infer this value
                        'expiration_date': None,  # TODO: infer this value
                        'confidentiality': None,  # TODO: infer this value
                        'languages': None,  # TODO: infer this value
                        'contents': [
                            {
                                'text': None,
                                'page': i + 1,
                                'section': None,  # TODO: infer this value
                                'text_type': None,  # TODO: infer this value
                                'languages': [],
                            }
                            for i in range(len(json_object))  # FIXME: tihs is wrong
                        ],
                    }
                    json.dump(organized_object, f, indent=4)

        elif file_path.suffix.lower() in ['.txt']:
            if args.no_overwrite and os.path.exists(out_path):
                print('--no_overwrite, skipping existing file:', out_path)
                return
            with open(file_path, 'r') as f:
                text = f.read()
            # only write if not empty:
            if len(text) > 5:
                with open(out_path, 'w') as f:
                    f.write(text)
        elif file_path.suffix.lower() in ['.xls', '.xlsx', '.csv']:
            # write as csv
            try:
                if args.no_overwrite and os.path.exists(out_path):
                    print('--no_overwrite, skipping existing file:', out_path)
                    return
                df = pd.read_csv(file_path)
                df.to_csv(out_path, index=False)
            except Exception:
                # sadly some CSV files are actually named as XLSX files
                if args.no_overwrite and os.path.exists(out_path):
                    print('--no_overwrite, skipping existing file:', out_path)
                    return
                df = pd.read_excel(file_path)
                out_path = out_path.replace(Path(out_path).suffix, '.csv')
                df.to_csv(out_path, index=False)
        else:
            # If the file type is not supported, print a warning message
            print(f'Warning: File type of {file_path.suffix} is not supported. Skipping file {file_path}.')
    except Exception as e:
        # If an error occurs during processing, print an error message
        print(f'Error: {e}. Skipping file {file_path}.')
        raise e


if __name__ == '__main__':
    file_paths = list(Path(args.data_root).rglob('**/*.*'))

    def init_main():
        with multiprocessing.Pool(args.processors) as pool:
            with tqdm(total=len(file_paths)) as pbar:
                for _ in pool.imap_unordered(process_file, zip(file_paths, [args.data_root] * len(file_paths))):
                    pbar.update()

    init_main()
