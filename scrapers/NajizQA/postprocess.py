from bs4 import BeautifulSoup
import re
import json


najizQA = json.load(open('./najiz_faq.json'))


def get_innder_text_from_HTML_string(html):
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()


def remove_double_spaces(text):
    return re.sub(' +', ' ', text.replace('\xa0', ' ')).strip()


benchmark = [
    {
        'question': remove_double_spaces(get_innder_text_from_HTML_string(x['question'])),
        'ground_truth': remove_double_spaces(get_innder_text_from_HTML_string(x['answer'])),
    }
    for x in najizQA
]

with open('benchmark.json', encoding='utf8') as f:
    json.dump(benchmark, f, indent=4, ensure_ascii=False)
