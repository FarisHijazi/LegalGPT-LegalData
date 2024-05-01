"""
scrapes https://new.najiz.sa/applications/landing/faq
"""

import requests
import json

# Fetch data
response = requests.get('https://new.najiz.sa/applications/landing/api/FaqQuestion/GetQuestions', headers={})

data = response.json()

category_ids = [z['categoryId'][0] for z in data]

response = requests.get('https://new.najiz.sa/applications/landing/api/FaqCategory/Get')
catmap = response.json()
id2label = {}
for item in catmap:
    id2label[item['id']] = item['name']

# write to file
with open('najiz_categories.json', 'w') as f:
    json.dump(id2label, f, indent=4)

for item in data:
    item['category'] = [id2label[x] for x in item['categoryId']]
with open('najiz_faq.json', 'w', encoding='utf8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
