# Scrape https://ncar.gov.sa

```sh
pip install -r requirements.txt


export NCAR_DATA_PATH="../../raw/Legal Data/المركز الوطني للوثائق والمحفوظات/rules-regulations"
# download pages_data.json, ar_i18n.json, all_releases_organizations.json
# create pages_data_populated.json
python scrape.py --use_cached \
    --output_dir "$NCAR_DATA_PATH"

# downloads PDFs using info in the JSONs
python download_pdfs.py \
    --pages_data_populated ./pages_data_populated.json \
    --output_dir "$NCAR_DATA_PATH/pdfs"

# outputs an interactive graph.html
python visualize_graph.py --pages_data_populated \
    "$NCAR_DATA_PATH/pages_data_populated.json" \
    --limit 500

```
