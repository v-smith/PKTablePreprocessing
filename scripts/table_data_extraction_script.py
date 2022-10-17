import jsonlines
from pk_tables.table_data_extraction import parse_xml_tables
from tqdm import tqdm


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


# read in json file
with jsonlines.open("../data/json/split_pk_pmcs_ner_dec2021/trial_ner_50.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

json_list_split = list(split(json_list, len(json_list)))

pkl_file_name = "table_hashes_trial_ner.pkl"
total_parsed_table_dict_list = []
for l in tqdm(json_list_split):
    parsed_table_dict_list = parse_xml_tables(l, pkl_file_name)
    total_parsed_table_dict_list.extend(parsed_table_dict_list)

# save as jsonl file
with jsonlines.open("../data/json/parsed_pk_pmcs_ner_dec2021/value_dicts/" + "parsed_trial_ner_50_value.jsonl",
                    mode='w') as writer:
    writer.write_all(total_parsed_table_dict_list)
