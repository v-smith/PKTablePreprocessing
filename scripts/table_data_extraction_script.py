import jsonlines
from pk_tables.table_data_extraction import parse_xml_tables
from tqdm import tqdm

print("hello")
# read in json file
with jsonlines.open("../data/json/split_pk_pmcs_ner_dec2021/remaining_ner.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


json_list_split = list(split(json_list, 8))

# parse the tables
# json_list1 = [x for x in json_list if x["text"] in ["PMC4833154 | Table 2 | DOI: 10.1111/bcp.12792", "PMC4833154 | Table 3 | DOI: 10.1111/bcp.12792", "PMC2151854 | Table 1 | DOI: 10.1186/cc5150"]]
# json_list = json_list[:10]

pkl_file_name = "table_hashes_remaining_ner.pkl"
total_parsed_table_dict_list = []
for l in tqdm(json_list_split):
    parsed_table_dict_list = parse_xml_tables(l, pkl_file_name)
    total_parsed_table_dict_list.extend(parsed_table_dict_list)

# save as jsonl file
with jsonlines.open("../data/json/parsed_pk_pmcs_ner_dec2021/" + "parsed_remaining_ner.jsonl", mode='w') as writer:
    writer.write_all(total_parsed_table_dict_list)

print("done")
a = 1
