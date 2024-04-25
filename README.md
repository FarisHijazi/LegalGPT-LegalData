# LegalData

This is a repo for collection and curation of the Arabic Saudi Legal Dataset as part of the LegalGPT project

## Download and setup

```bash
git clone https://github.com/a-fhijazi_thq/LegalData.git
cd LegalData
pip install -r requirements.txt

sudo apt install poppler-utils -y

# setup the .env file
# MAKE SURE TO OPEN IT AND SET THE API KEYS
cp .env.example .env
```

## Data Collection

### DVC setup

```bash
# create a remote if you don't have one:
az storage container create --name $AZURE_STORAGE_ACCOUNT --account-name $AZURE_STORAGE_ACCOUNT --account-key $AZURE_STORAGE_KEY

# find the endpoint and key in the .env file
dvc remote add -d azure_legalgpt azure://$AZURE_STORAGE_ACCOUNT/legalgpt
dvc remote modify azure_legalgpt connection_string "AccountName=$AZURE_STORAGE_ACCOUNT;SharedAccessSignature=$AZURE_STORAGE_SAS_TOKEN"

```

All the documentation of the data sources can be found in this [Notion page](https://www.notion.so/Open-data-repository-fffd5adfbce74a738a0243eb02fdd62f?pvs=4)

## Downloading the data:


After setting up `dvc`

```bash
source .env  # make sure to set the env variables
dvc pull
```

The data was scraped from June 22, 2023 - June 25, 2023

<details>
<summary>⚠️ Old method of downloading data: Download it from pcloud</summary>

Manually download raw data in a zip file

<https://u.pcloud.link/publink/show?code=XZEKEF0ZERSjlk2Tp6zPjwoK1xgL6JqAJGkk>

unzip it into the `data/raw/` folder

</details>

## Unified Schema

The schema represents a single regulation (تعميم او تشريع)

The most important field is: `regulation_schema['contents'][i]['text']` which contains the extracted text from the source file

```js
regulation_schema = {
    "preprocess_script_git_hash": "1234567890abcdef", // the git hash of the script that generated this file
    "schema_version": "1.0",
    "source_entity": "ministry of justice",     // جهة الإصدار
    "origin_url": "http://example.com",
    "serial_number": "123456",
    "original_file_path": "path/to/file.pdf",
    "document_type": "regulation",              // نوع الوثيقة
    "circular_topics": ["category",],           // موضوع التعميم
    "circular_number": "123",                   // رقم التعميم
    "title": "Legal Document Title",
    "issue_date": "2022-11-01",
    "effective_date": "2022-12-01",
    "expiration_date": "2023-12-31",
    "confidentiality": "public",
    "languages": ["ar", "en"],
    "contents": [
        // these are the extracted texts from the PDF
        {"text": "extracted text", "page": 1, "section": "Introduction", "text_type": "paragraph", "languages": ["ar"]},
        {"text": "extracted text", "page": 2, "section": "Section 1", "text_type": "bullet_point", "languages": ["ar"]}
    ],
}
```

## Processing

Use the `convert.py` processes the raw files and outputs them in the `processed/` folder.
It does the following:

Dataset preprocessor

this script processes lots of random unstructured files from a dataset

This script operates differently depending on the file type, but the goal is to output a unified JSON format.

- `.pdf`: convert pages to images -> run OCR on each page -> convert `regulation_schema`
- `.xls, .xlsx, .csv`: convert `regulation_schema`
- `.txt`: convert `regulation_schema`
- `.json`: convert `regulation_schema`


```bash

python convert.py data/raw --out_dir data/processed --whitelist .pdf --ocr azure --processors 50

#TODO: add step to take this unified JSON and fill the missing fields by reading the content with an LLM
```
