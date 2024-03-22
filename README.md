# LegalData

This is a repo for collection and curation of the Arabic Saudi Legal Dataset as part of the LegalGPT project

## Download and setup

```bash
git clone https://github.com/a-fhijazi_thq/LegalData.git
cd LegalData
pip install -r requirements.txt

sudo apt install poppler-utils -y

```

## Data Collection

All the documentation of the data sources can be found in this [Notion page](https://www.notion.so/Open-data-repository-fffd5adfbce74a738a0243eb02fdd62f?pvs=4)

## Downloading the data:

Download it from pcloud:

<https://u.pcloud.link/publink/show?code=XZEKEF0ZERSjlk2Tp6zPjwoK1xgL6JqAJGkk>

unzip it into the `data/raw/` folder

## Processing

We need to use OCR for extracting the PDFs

Use the `extract.py` script to extract the text from the PDFs


```bash
python preprocess.py data/raw --output_dir data/processed
```

