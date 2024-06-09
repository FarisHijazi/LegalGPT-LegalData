To run QA benchmarking, you will have to populate the ./benchmarkQA/utils.py with the language model APIs as follows:

```python
...
llms['Cohere-command-r-kscml'] = OpenAI(
    engine='Cohere-command-r-kscml',
    api_base='https://...',
    api_key='...',
)
...
```


```sh
python benchmark.py
```

