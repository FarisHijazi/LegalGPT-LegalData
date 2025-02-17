"""
Dataset preprocessor

this script processes lots of random unstructured files from a dataset

This script operates differently depending on the file type, but the goal is to output a unified JSON format.

- .pdf: convert pages to images -> run OCR on each page -> convert to unified JSON
- .txt: convert to unified JSON
- .json: convert to unified JSON

"""

import argparse
import fnmatch
import io
import json
import logging
import mimetypes
import multiprocessing
import os
import re
import subprocess
import time
from io import BytesIO
from pathlib import Path

import backoff
import openai
import pdf2image
from dotenv import load_dotenv
from google.cloud.vision_v1.types.image_annotator import AnnotateImageResponse
from joblib import Memory
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(console_handler)


load_dotenv()

parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_root', type=Path, help='Path to data root directory')
parser.add_argument(
    '-o',
    '--out_dir',
    default='./processed/',
    type=Path,
    help='Path to save the extracted text',
)
parser.add_argument(
    '--overwrite',
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
parser.add_argument(
    '--danger_skip_hash_check',
    default=False,
    action='store_true',
)
parser.add_argument(
    '--pdf_save_intermediate',
    default=False,
    action='store_true',
    help='Save individual PDF images',
)
parser.add_argument(
    '--ocr_save_annotated_images',
    default=False,
    action='store_true',
    help='Save OCR image annotations',
)
# add argument list to whitelist specific file extensions
parser.add_argument(
    '--ocr',
    default='google',
    choices=['google', 'azure', 'azuredocumentanalysis'],
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

BLACKLIST_GLOB = [
    '**/المنصة القضائية العلمية/**',
]


if args.ocr == 'google':
    from google.cloud import vision_v1

    # Supported mime_type: application/pdf, image/tiff, image/gif
    google_vision_client = vision_v1.ImageAnnotatorClient()
elif args.ocr == 'azure':
    from azure.cognitiveservices.vision.computervision import \
        ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials

    azure_computervision_client = ComputerVisionClient(
        os.environ['VISION_ENDPOINT'],
        CognitiveServicesCredentials(os.environ['VISION_KEY']),
    )
elif args.ocr == 'azuredocumentanalysis':
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential

    # Initialize the DocumentAnalysisClient
    document_analysis_client = DocumentAnalysisClient(
        endpoint=os.environ['FORM_RECOGNIZER_ENDPOINT'],
        credential=AzureKeyCredential(os.environ['FORM_RECOGNIZER_KEY']),
    )
else:
    raise ValueError('Invalid OCR provider')


def read_file_content(file_path):
    with open(file_path, 'rb', encoding='utf8') as f:
        return f.read()


def get_git_hash():
    """get git hash of the current commit for this file convert.py"""
    # check if there are any changes to the current file, if so, raise an exception
    if subprocess.check_output(['git', 'status', '--porcelain', __file__]) and not args.danger_skip_hash_check:
        raise Exception(
            'ERROR: There are uncommitted changes to this file, please commit them before running this script, this is to verify script and data integrity.'
            '\nIf you are sure you want to run this script with uncommitted changes, use the --danger_skip_hash_check flag.'
        )
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')


# get git hash:
GIT_HASH = get_git_hash()


# Create a memory object for caching
memory = Memory('./cachedir', verbose=0)


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
        raise Exception(f"Failed to guess the mime type of {image}")

    if mime_type != 'image/tiff':
        logger.warning('Converting PNG to TIFF')
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


def google_ocr_annotate_and_save(image_path, image_response, file_content=None):
    # Get the image
    image_path = Path(image_path)
    if file_content is None:
        image = Image.open(BytesIO(file_content))
    else:
        image = Image.open(image_path)

    if not args.ocr_save_annotated_images:
        return

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Draw rectangles for each block
    for page in image_response.full_text_annotation.pages:
        for block in page.blocks:
            try:
                vertices = [(vertex.x, vertex.y) for vertex in block.bounding_box.vertices]
                draw.rectangle([vertices[0], vertices[2]], outline='red')
            except Exception as e:
                logger.error(f'Failed to draw rectangle for block, error: {e}')

    # Create the new file name
    new_file_name = image_path.parent / f"{image_path.stem}.annotated{image_path.suffix}"

    # Save the image
    logger.info(f'saved to "{new_file_name}"')
    image.save(new_file_name)


if not args.no_ocr_cache:
    google_ocr_single_image = memory.cache(google_ocr_single_image)


def azure_ocr_single_image(image: bytes):
    """
    OCR: Read File using the Read API, extract text - remote
    This example will extract text in an image, then print results, line by line.
    This API call can also extract handwriting style text (not shown).
    """

    read_response = azure_computervision_client.read_in_stream(io.BytesIO(image), raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers['Operation-Location']
    # Grab the ID from the URL
    operation_id = read_operation_location.split('/')[-1]

    # Call the "GET" API and wait for it to retrieve the results
    while True:
        read_result = azure_computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(0.5)

    # Print the detected text, line by line
    # if read_result.status == OperationStatusCodes.succeeded:
    #     for text_result in read_result.analyze_result.read_results:
    #         for line in text_result.lines:
    #             print(line.text)
    #             print(line.bounding_box)

    return read_result


def azuredocumentanalysis_ocr_single_image(image: bytes):
    """
    OCR: Read File using the Read API, extract text - remote
    This example will extract text in an image, then print results, line by line.
    This API call can also extract handwriting style text (not shown).
    """
    read_response = document_analysis_client.begin_analyze_document('prebuilt-document', io.BytesIO(image), locale='ar')
    result = read_response.result()
    return result

    # # Get the operation location (URL with an ID at the end) from the response
    # read_operation_location = read_response.headers['Operation-Location']
    # # Grab the ID from the URL
    # operation_id = read_operation_location.split('/')[-1]

    # # Call the "GET" API and wait for it to retrieve the results
    # while True:
    #     read_result = azure_computervision_client.get_read_result(operation_id)
    #     if read_result.status not in ['notStarted', 'running']:
    #         break
    #     time.sleep(1)

    # # Print the detected text, line by line
    # # if read_result.status == OperationStatusCodes.succeeded:
    # #     for text_result in read_result.analyze_result.read_results:
    # #         for line in text_result.lines:
    # #             print(line.text)
    # #             print(line.bounding_box)

    # return read_result


def azure_ocr_annotate_and_save(image_path, read_result, file_content=None):
    # Get the image
    image_path = Path(image_path)

    # Create the new file name
    new_file_name = image_path.parent / f"{image_path.stem}.annotated{image_path.suffix}"

    if not args.ocr_save_annotated_images:
        return

    if file_content:
        image = Image.open(BytesIO(file_content))
    else:
        image = Image.open(image_path)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Loop over each line in the result
    for text_result in read_result['analyze_result']['read_results']:
        for line in text_result['lines']:
            # Get the bounding box coordinates
            bounding_box = line['bounding_box']

            # Draw rectangle
            draw.rectangle(
                [
                    (bounding_box[0], bounding_box[1]),
                    (bounding_box[4], bounding_box[5]),
                ],
                outline='red',
            )

    # Save the image
    image.save(new_file_name)
    logger.info(f'azure_ocr_annotate_and_save() saved to "{new_file_name}"')


def azuredocumentanalysis_ocr_annotate_and_save(image_path, read_result, file_content=None):
    """

    example read_result:

    ```json
    {
        "api_version": "2023-07-31",
        "model_id": "prebuilt-document",
        "content": "...",
        "languages": [],
        "pages": [
            {
                "page_number": 1,
                "angle": -0.30979999899864197,
                "width": 4603.0,
                "height": 6481.0,
                "unit": "pixel",
                "lines": [
                    {
                        "content": "\u25cc\u0650\u0646\u0652 \u0623\u064e \u0644\u064e \u0627\u0644\u064e",
                        "polygon": [
                            {
                                "x": 1978.0,
                                "y": 122.0
                            },
                            {
                                "x": 2578.0,
                                "y": 92.0
                            },
                            {
                                "x": 2604.0,
                                "y": 325.0
                            },
                            {
                                "x": 1987.0,
                                "y": 346.0
                            }
                        ],
                        "spans": [
                            {
                                "offset": 0,
                                "length": 14
                            }
                        ]
                    },
                    {
                        "content": "\u0627\u0644\u0645\u064e\u0644\u0650\u0643\u064e\u0629\u064f \u0627\u0644\u0639\u064e\u0651\u0629 \u0627\u0644\u0633\u064e\u0651\u0639\u064f\u0648\u0631\u064a\u064e\u0651\u0629",
                        "polygon": [
                            {
                                "x": 3226.0,
                                "y": 450.0
                            },
                            {
                                "x": 4344.0,
                                "y": 432.0
                            },
                            {
                                "x": 4348.0,
                                "y": 653.0
                            },
                            {
                                "x": 3226.0,
                                "y": 663.0
                            }
                        ],
                        "spans": [
                            {
                                "offset": 15,
                                "length": 31
                            }
                        ]
                    },
                    ...
                ],
                "words": [
                    {
                        "content": "\u25cc\u0650\u0646\u0652",
                        "polygon": [
                            {
                                "x": 2437.0,
                                "y": 98.0
                            },
                            {
                                "x": 2585.0,
                                "y": 92.0
                            },
                            {
                                "x": 2595.0,
                                "y": 325.0
                            },
                            {
                                "x": 2447.0,
                                "y": 332.0
                            }
                        ],
                        "span": {
                            "offset": 0,
                            "length": 4
                        },
                        "confidence": 0.094
                    },
                    ...
                ],
                "selection_marks": [],
                "spans": [
                    {
                        "offset": 0,
                        "length": 367
                    }
                ],
                "barcodes": [],
                "formulas": []
            }
        ],
        "paragraphs": [
            {
                "role": null,
                "content": "\u25cc\u0650\u0646\u0652 \u0623\u064e \u0644\u064e \u0627\u0644\u064e",
                "bounding_regions": [
                    {
                        "page_number": 1,
                        "polygon": [
                            {
                                "x": 1978.0,
                                "y": 116.0
                            },
                            {
                                "x": 2595.0,
                                "y": 91.0
                            },
                            {
                                "x": 2604.0,
                                "y": 325.0
                            },
                            {
                                "x": 1987.0,
                                "y": 350.0
                            }
                        ]
                    }
                ],
                "spans": [
                    {
                        "offset": 0,
                        "length": 14
                    }
                ]
            },
            ...
        ],
        "tables": [],
        "key_value_pairs": [],
        "styles": [
            {
                "is_handwritten": true,
                "similar_font_family": null,
                "font_style": null,
                "font_weight": null,
                "color": null,
                "background_color": null,
                "spans": [
                    {
                        "offset": 0,
                        "length": 14
                    }
                ],
                "confidence": 0.0
            }
        ],
        "documents": []
    }
    ```

    """
    # Get the image
    image_path = Path(image_path)

    # Create the new file name
    new_file_name = image_path.parent / f"{image_path.stem}.annotated{image_path.suffix}"

    if not args.ocr_save_annotated_images:
        return

    if file_content:
        image = Image.open(BytesIO(file_content))
    else:
        image = Image.open(image_path)

    # Create a draw object
    draw = ImageDraw.Draw(image)

    # Loop over each page in the result
    for page in read_result['pages']:
        # Draw lines
        for line in page['lines']:
            # Get the polygon points
            polygon_points = [(point['x'], point['y']) for point in line['polygon']]

            # Draw polygon around text
            for i in range(len(polygon_points)):
                start = polygon_points[i]
                end = polygon_points[(i + 1) % len(polygon_points)]
                draw.line([start, end], fill='red', width=2)

            # Optionally draw the text content above the box
            draw.text((polygon_points[0][0], polygon_points[0][1] - 10), line['content'], fill='blue')

        # Draw tables if present
        if 'tables' in page:
            for table in page['tables']:
                for cell in table['cells']:
                    polygon_points = [(point['x'], point['y']) for point in cell['polygon']]
                    for i in range(len(polygon_points)):
                        start = polygon_points[i]
                        end = polygon_points[(i + 1) % len(polygon_points)]
                        draw.line([start, end], fill='green', width=2)

    # Save the image
    image.save(new_file_name)
    logger.info(f'azuredocumentanalysis_ocr_annotate_and_save() saved to "{new_file_name}"')


if not args.no_ocr_cache:
    azure_ocr_single_image = memory.cache(azure_ocr_single_image)


def PIL_to_bytes(image, format='TIFF'):
    # a hack
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=format)
    return byte_arr.getvalue()


# === end of OCR functions ===


@memory.cache
def llm_json(prompt: str, model='gpt-4o-mini'):
    openai.api_key = os.getenv('OPENAI_API_KEY')

    completion = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        messages=[
            {
                'role': 'system',
                'content': 'Return JSON only, using the following keys: {issue_date, effective_date, expiration_date, circular_number,}',
            },
            {'role': 'user', 'content': prompt},
        ],
        model=model,
        response_format={'type': 'json_object'},
    )
    return json.loads(completion.choices[0].message['content'])


def split_on_one_or_more_white_characters(string):
    import re

    return [s.strip() for s in re.split(r'[\n\r]+', string)]


def parse_date_and_circular_number(text):
    # Extract the circular number
    circular_number = re.search(r'رقم (\d+) ', text)
    if circular_number:
        circular_number = circular_number.group(1)

    # Extract the issue date
    issue_date = re.search(r'(\d+/?\d+/?\d+ه)', text)
    if issue_date:
        issue_date = issue_date.group(1)
    return {'circular_number': circular_number, 'issue_date': issue_date}


# =====


def process_json(obj, file_path='') -> list[dict]:
    # Initialize result list
    results = []

    # Check the type of obj and process accordingly
    if isinstance(obj, dict):

        # Return a dictionary with pre-defined keys and values
        result = {
            'preprocess_script_git_hash': GIT_HASH,
            'untrustworthy_git_hash': args.danger_skip_hash_check,
            'schema_version': '1.0',
            'source_entity': None,
            'origin_url': None,
            'serial_number': None,
            'original_file_path': str(file_path),
            'document_type': None,
            'circular_topic': None,
            'circular_number': None,
            'title': None,
            'issue_date': None,
            'effective_date': None,
            'expiration_date': None,
            'confidentiality': None,
            'languages': None,
            'contents': [
                {
                    'text': None,  # TODO: replace with actual text
                    'page': None,
                    'section': None,
                    'text_type': None,
                    'languages': ['arabic'],
                }
            ],
        }

        # TODO: aggregate multiple strings ( وعنوان الحكم نص التعميم والمختصر والاستئناف ونص احكم ونص التعميم)
        result['contents'][0]['text'] = ''
        text_keys = [
            'text',
            'عنوان الحكم',
            'بيانات الحكم',
            'نص التعميم',
            'نص الحكم',
            'الاستئناف',
        ]
        for key in text_keys:
            if key in obj:
                result['contents'][0]['text'] += '## ' + key + ':\n---\n\n' + obj[key].strip() + '\n\n'
        result['contents'][0]['text'] = result['contents'][0]['text'].strip()

        if '(. . .' in result['contents'][0]['text'] or '(...' in result['contents'][0]['text']:
            logger.warning('text contains incomplete text')
            result['contents'][0]['text'] = ''

        if 'التصنيف' in obj:
            result['circular_topics'] = split_on_one_or_more_white_characters(obj['التصنيف'].strip())
        if 'موضوع التعميم' in obj:
            result['circular_topics'] = split_on_one_or_more_white_characters(obj['موضوع التعميم'].strip())

        result['circular_number'] = obj.get('رقم التعميم')
        result['issue_date'] = obj.get('تاريخ التعميم')
        result['origin_url'] = obj.get('url')

        if 'بيانات الحكم' in obj:
            o = parse_date_and_circular_number(obj['بيانات الحكم'])
            result['circular_number'] = o['circular_number']
            result['issue_date'] = o['issue_date']

        if 'عنوان الحكم' in obj:
            result['title'] = obj['عنوان الحكم']

        results.append(result)
    elif isinstance(obj, list):
        for j in obj:
            results.extend(process_json(j, file_path))
    else:
        if type(obj) is str and len(obj) > 50:
            logger.warning(f'string passed instead of json object, converting... "{str(obj)}"')
            results.append(process_json({'text': str(obj)}))
        else:
            logger.warning(f'skipping file: "{file_path}"')
    # TODO: deal with strings

    return results


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def process_image(i, image, file_path, out_path):
    logger.debug(f'processing image {i+1}')
    image_path = os.path.join(out_path, f"page_{str(i+1).zfill(3)}.tiff")
    if not args.overwrite and os.path.exists(image_path):
        logger.debug(f'--overwrite, skipping existing file: "{image_path}"')
        return

    output_path_organized = image_path + '.organized.json'

    logger.debug(f'saving image to "{image_path}"')
    if args.pdf_save_intermediate:
        image.save(image_path, 'TIFF')

    content = PIL_to_bytes(image, format='TIFF')

    if args.ocr == 'google':
        image_response = google_ocr_single_image(content)
        from google.protobuf.json_format import MessageToDict

        google_ocr_annotation_outpath = image_path + '.google_ocr_annotation.json'
        if args.overwrite or not os.path.isfile(google_ocr_annotation_outpath):
            with open(google_ocr_annotation_outpath, 'w', encoding='utf8') as f:
                json.dump(MessageToDict(image_response), f, ensure_ascii=False, indent=4)
        try:
            if args.ocr_save_annotated_images:
                google_ocr_annotate_and_save(image_path, image_response, file_content=content)
        except Exception as e:
            logger.error(f'Failed to annotate image "{image_path}", error: {e}')
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
    elif args.ocr == 'azure':
        azure_ocr_annotation_outpath = image_path + '.azure_ocr_annotation.json'
        if not args.overwrite and os.path.isfile(azure_ocr_annotation_outpath):
            with open(azure_ocr_annotation_outpath, 'r', encoding='utf8') as f:
                read_result = json.load(f)
                # print("using existing azure ocr result", azure_ocr_annotation_outpath)
        else:
            logger.info(f'azure ocr annotation not found, running new OCR "{azure_ocr_annotation_outpath}"')
            read_result = azure_ocr_single_image(content).as_dict()
            with open(azure_ocr_annotation_outpath, 'w', encoding='utf8') as f:
                json.dump(read_result, f, ensure_ascii=False, indent=4)

        try:
            if args.ocr_save_annotated_images:
                azure_ocr_annotate_and_save(image_path, read_result, file_content=content)
        except Exception as e:
            logger.error(f'Failed to annotate image "{image_path}", error: {e}')
        # TODO: double check this
        contents = [
            {
                'text': line.text,
                'page': i + 1,
                'section': None,
                'text_type': None,
                'languages': None,
            }
            for i, text_result in enumerate(read_result['analyze_result']['read_results'])
            for line in text_result['lines']
        ]
    elif args.ocr == 'azuredocumentanalysis':
        azuredocumentanalysis_ocr_annotation_outpath = image_path + '.azuredocumentanalysis_ocr_annotation.json'
        if not args.overwrite and os.path.isfile(azuredocumentanalysis_ocr_annotation_outpath):
            logger.debug(f'found existing ocr result "{azuredocumentanalysis_ocr_annotation_outpath}"')
            if os.path.isfile(output_path_organized):
                logger.debug(f'found existing organized result, skipping.. "{output_path_organized}"')
                return

            with open(
                azuredocumentanalysis_ocr_annotation_outpath,
                'r',
                encoding='utf8',
            ) as f:
                read_result = json.load(f)
        else:
            logger.debug(f'azuredocumentanalysis ocr annotation not found, running new OCR "{azuredocumentanalysis_ocr_annotation_outpath}"')
            read_result = azuredocumentanalysis_ocr_single_image(content).to_dict()
            time.sleep(0.1)

            with open(
                azuredocumentanalysis_ocr_annotation_outpath,
                'w',
                encoding='utf8',
            ) as f:
                json.dump(read_result, f, ensure_ascii=False, indent=4)
                logger.info(f'wrote to "{azuredocumentanalysis_ocr_annotation_outpath}"')

        if args.ocr_save_annotated_images:
            azuredocumentanalysis_ocr_annotate_and_save(image_path, read_result, file_content=content)

        contents = [
            {
                'text': paragraph['content'],
                'page': paragraph['bounding_regions'][0]['page_number'],
                'section': None,
                'text_type': 'paragraph',
                'languages': (read_result['languages'] if read_result['languages'] else ['ar']),
            }
            for paragraph in read_result['paragraphs']
        ]

        example_kvp = [
            {
                'key': {
                    'content': 'المملكة:',
                    'bounding_regions': [
                        {
                            'page_number': 1,
                            'polygon': [
                                {'x': 1470.0, 'y': 504.0},
                                {'x': 1558.0, 'y': 504.0},
                                {'x': 1558.0, 'y': 538.0},
                                {'x': 1470.0, 'y': 538.0},
                            ],
                        }
                    ],
                    'spans': [{'offset': 178, 'length': 8}],
                },
                'value': None,
                'confidence': 0.145,
            },
        ]

        # Add key-value pairs and tables
        contents.extend(
            [
                {
                    'text': kvp['key']['content'] + ': ' + (kvp['value']['content'] if kvp['value'] else ''),
                    'page': kvp['key']['bounding_regions'][0]['page_number'],
                    'section': None,
                    'text_type': 'key_value_pair',
                    'languages': (read_result['languages'] if read_result['languages'] else ['ar']),
                }
                for kvp in read_result['key_value_pairs']
            ]
        )

        # table
        contents.extend(
            [
                {
                    'text': str(table),
                    'page': table['bounding_regions'][0]['page_number'],
                    'section': None,
                    'text_type': 'table',
                    'languages': (read_result['languages'] if read_result['languages'] else ['ar']),
                }
                for table in read_result['tables']
            ]
        )
    else:
        raise ValueError('Invalid OCR provider')

    with open(output_path_organized, 'w', encoding='utf8') as f:
        organized_object = {
            'preprocess_script_git_hash': GIT_HASH,
            'untrustworthy_git_hash': args.danger_skip_hash_check,
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
        json.dump(organized_object, f, indent=4, ensure_ascii=False)
        logger.info(f'wrote to "{output_path_organized}"')


def try_except_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")

    return wrapper


# @try_except_decorator
def process_file(file_path_inpath):
    file_path, inpath = file_path_inpath
    relative_path = os.path.relpath(file_path, inpath)
    out_path = os.path.join(args.out_dir, relative_path)

    if any([fnmatch.filter([str(file_path)], pattern) for pattern in BLACKLIST_GLOB]):
        logger.info(f'File is blacklisted. Skipping file. "{file_path}"')
        return

    # TODO: delete empty folders at the end
    os.makedirs(out_path, exist_ok=True)

    # TODO: maybe make stage 1 to be JSONs and then stage 2 is the OCR and the
    if file_path.suffix.lower() == '.pdf':
        try:
            if not args.overwrite and os.path.exists(os.path.join(out_path, f"page_001.tiff.organized.json")):
                logger.debug(f'--overwrite, skipping existing organized files for: "{file_path}"')
                return
            images: list[Image.Image] = pdf2image.convert_from_path(file_path)  # heavy work
        except pdf2image.exceptions.PDFPageCountError:
            logger.error(f'Corrupted PDF, DELETING: "{file_path}"')
            os.remove(file_path)
            return

        # # Create a ThreadPoolExecutor
        # with ThreadPoolExecutor(max_workers=1) as executor:  # setting this to >1 might cause joblib issues
        #     # Use map to execute the function in parallel
        #     # executor.map(process_image, zip(range(len(images)), images, [out_path]*len(images)))
        #     executor.
        list(map(process_image, range(len(images)), images, [file_path] * len(images), [out_path] * len(images)))
    elif file_path.suffix.lower() in ['.json']:
        if not args.overwrite and os.path.exists(out_path):
            logger.info(f'--overwrite, skipping existing file: "{out_path}"')
            return
        with open(file_path, 'r', encoding='utf8') as f:
            data = f.read()
        # only write if not empty:
        if len(data) > 5:
            json_object = json.loads(data)
            if type(json_object) is not dict and not type(json_object[0]) is dict:
                logger.warning(f'File is not a valid JSON file. Skipping file. "{file_path}"')
                return
            processed_jsons = process_json(json_object, file_path)

            for i, processed_json in enumerate(processed_jsons):
                file_out_path = os.path.join(out_path, f"{str(i+1).zfill(3)}.organized.json")
                with open(file_out_path, 'w', encoding='utf8') as f:
                    json.dump(processed_json, f, indent=4, ensure_ascii=False)
    elif file_path.suffix.lower() in ['.txt']:
        if not args.overwrite and os.path.exists(out_path):
            logger.info(f'--overwrite, skipping existing file: "{out_path}"')
            return
        with open(file_path, 'r', encoding='utf8') as f:
            text = f.read()
        # only write if not empty:
        if len(text) < 50:
            logger.warning(f'File is empty. Skipping file. "{file_path}"')
            return

        processed_json = process_json({'text': text}, file_path)[0]
        file_out_path = os.path.join(out_path, f"{os.path.basename(file_path)}.organized.json")
        with open(file_out_path, 'w', encoding='utf8') as f:
            json.dump(processed_json, f, indent=4, ensure_ascii=False)
    else:
        # If the file type is not supported, print a warning message
        logger.warning(f"File type of {file_path.suffix} is not supported. Skipping file {file_path}.")


if __name__ == '__main__':
    # process_file((Path("./raw/Legal Data/المركز الوطني للوثائق والمحفوظات/rules-regulations/pdfs/5a951ac514cb141cd2573a56e2c9a8a012ffd983226362988766e547ba98bd12.pdf"), args.data_root))
    file_paths = list(Path(args.data_root).rglob('**/*.*'))
    # reduce to whitelisted
    file_paths = [x for x in file_paths if x.suffix in args.whitelist and x.is_file()]

    def init_main():
        with multiprocessing.Pool(args.processors) as pool:
            map_fn = pool.imap_unordered if args.processors > 1 else map
            with tqdm(total=len(file_paths)) as pbar:
                for _ in map_fn(process_file, zip(file_paths, [args.data_root] * len(file_paths))):
                    pbar.update()

    init_main()
