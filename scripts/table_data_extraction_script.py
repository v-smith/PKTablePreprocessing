import jsonlines
from pk_tables.table_data_extraction import parse_xml_tables

# read in json file
with jsonlines.open("../data/json/split_pk_pmcs_ner_dec2021/trial_ner_50.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

pkl_file_name = "table_hashes_trial_ner_50.pkl"
# parse the tables
#json_list1 = [x for x in json_list if x["text"] in ["PMC4833154 | Table 2 | DOI: 10.1111/bcp.12792", "PMC4833154 | Table 3 | DOI: 10.1111/bcp.12792", "PMC2151854 | Table 1 | DOI: 10.1186/cc5150"]]
parsed_table_dict_list = parse_xml_tables(json_list, pkl_file_name)

# save as jsonl file
with jsonlines.open("../data/json/parsed_pk_pmcs_ner_dec2021/" + "parsed_trial_ner_50.jsonl", mode='w') as writer:
    writer.write_all(parsed_table_dict_list)

print("done")
a = 1
