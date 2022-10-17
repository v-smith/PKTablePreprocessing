# Run Prodigy Locally

Installation instructions - from command line 

Confirm python is installed 
```
python
```
To install python (skip if not needed)
```
https://www.python.org/downloads/windows/
```
Confirm pip is installed 
```
pip help
```

To install pip (skip if not needed)
```
python get-pip.py
#verify installation
pip -V
```

To Install all python packages needed 
```
pip install .
```

Install prodigy
```
pip install <prodigy-dir>/prodigy-<version>.whl 
```
## Run recipe locally  
```
python -m prodigy label-json tables_trial_labels data/json/test_100.jsonl -F recipes/label-json.py
python -m prodigy table-ner tables_trial_labels blank:en data/parsed_table_jsons/parsed_table_dicts_html.jsonl -F recipes/table-ner.py
python -m prodigy table-ner tables_trial_labels blank:en data/parsed_table_jsons/parsed_table_dicts_html_styled.jsonl -F recipes/table-ner.py --highlight-chars --label 1,2,3,4,5,6,7,8,9,10,11,12,13,14
#test one 
python -m prodigy ner.manual tables_trial_labels  data/parsed_table_jsons/parsed_table_dicts_html.jsonl --label PK,PD --highlight-chars
```
## Get annotations out 

```
python -m prodigy db-out tables_trial_labels > ./output_annotations.jsonl
```


