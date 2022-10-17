import jsonlines

with jsonlines.open("../data/json/cell_entities/parsed_val_ner_entities.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

with open('../data/json/relevant_ids/parsed_val_ner__ids_distilbert.txt', 'r') as readfile:
    relevant_ids = [line.strip() for line in readfile]

relevant_list = [item for item in json_list if item["table_id"] in relevant_ids]
relevant_list_for_class = [item for item in relevant_list if item["col"] == "na"]

not_relevant_list = [item for item in json_list if item["table_id"] not in relevant_ids]
not_relevant_for_class = [item for item in not_relevant_list if item["col"] == "na"]

assert (len(relevant_list) + len(not_relevant_list)) == len(json_list)

with jsonlines.open("../data/json/pk_tablesclass_data/relevant/" + "parsed_val_ner__relevant_forclass.jsonl", mode='w') as writer:
    writer.write_all(relevant_list_for_class)

with jsonlines.open("../data/json/pk_tablesclass_data/not_rel/" + "parsed_val_ner__NotRelevant_forclass.jsonl", mode='w') as writer:
    writer.write_all(not_relevant_for_class)


