# Scrape https://ncar.gov.sa

```sh
pip install -r requirements.txt

python scrape.py --use_cached
# will download pages_data.json, ar_i18n.json, all_releases_organizations.json
# will create pages_data_populated.json

python download_pdfs.py \
    --pages_data_populated ./pages_data_populated.json \
    --output_dir "../../data/raw/Legal Data/المركز الوطني للوثائق والمحفوظات/rules-regulations/pdfs"

python visualize_graph.py --pages_data_populated ./pages_data_populated.json --limit 500
# outputs an interactive graph.html

```