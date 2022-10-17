import jsonlines
import random

#set random seed
random.seed(10)

# read in json file
with jsonlines.open("../data/json/pk_pmcs_ner_dec2021/selected_pk_tables.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

random.shuffle(json_list)

trial_set = json_list[0:50]
train_set = json_list[50:2050]
test_set = json_list[2050:3050]
val_set = json_list[3050:4050]
remaining_set = json_list[4050:]

#write to file
with jsonlines.open("../data/json/parsed_table_jsons/trial_ner_50.jsonl", mode='w') as writer:
    writer.write_all(trial_set)

with jsonlines.open("../data/json/split_pk_pmcs_ner_dec2021/train_ner_2000.jsonl", mode='w') as writer:
    writer.write_all(train_set)

with jsonlines.open("../data/json/split_pk_pmcs_ner_dec2021/test_ner_1000.jsonl", mode='w') as writer:
    writer.write_all(test_set)

with jsonlines.open("../data/json/split_pk_pmcs_ner_dec2021/val_ner_1000.jsonl", mode='w') as writer:
    writer.write_all(val_set)

#write to file
with jsonlines.open("../data/json/split_pk_pmcs_ner_dec2021/remaining_ner.jsonl", mode='w') as writer:
    writer.write_all(remaining_set)

print("complete")
a=1