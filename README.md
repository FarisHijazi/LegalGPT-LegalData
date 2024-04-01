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


<!-- 
sudo apt install imagemagick -y
sudo cp /etc/ImageMagick-6/policy.xml /etc/ImageMagick-6/policy.xml.backup

sudo sed -i 's/rights="none"/rights="read|write"/' /etc/ImageMagick-*/policy.xml


srcext=pdf; find data/raw/ -type f -name "*.$srcext" -print0 | xargs -0 -P $(nproc --all) -I{} sh -c 'convert -density 300 "$1" -quality 100 "${1%.*}_page_%03d.png" | echo "" ' -- {} | tqdm --unit .$srcext --total $(find data/raw/ -type f -name "*.$srcext" | wc -l) > /dev/null
 -->

```bash

python convert.py data/raw --out_dir data/01_converted
```

