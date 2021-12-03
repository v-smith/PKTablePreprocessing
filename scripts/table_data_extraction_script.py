import jsonlines
from pk_tables.table_data_extraction import parse_xml_tables, hash_html_tables
import pandas as pd
import numpy as np
from IPython.display import HTML, Javascript
import pickle

# read in json file
with jsonlines.open("../data/json/pk_pmcs_ner_dec2021/selected_pk_tables.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

# select the first 10 entries for testing
json_list = json_list[6:8]
#json_list = [x for x in json_list if x["text"] == "PMC4747995 | Table 1 | DOI: 10.1007/s00280-015-2955-9"]

# parse the tables
parsed_table_dict_list = parse_xml_tables(json_list)

# save as jsonl file

#TODO: SPLIT THE DATA
#training and val set=> 2000
with jsonlines.open("../data/json/parsed_table_jsons/" + "table_ner_trial.jsonl", mode='w') as writer:
    writer.write_all(parsed_table_dict_list)

#test=> 1000
#with jsonlines.open("../data/parsed_table_jsons/" + "table_ner_test.jsonl", mode='w') as writer:
    #writer.write_all(parsed_table_dict_list)
a = 1
