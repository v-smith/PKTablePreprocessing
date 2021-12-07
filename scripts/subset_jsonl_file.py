import jsonlines
import random
#from sklearn.model_selection import train_test_split
#import numpy

# read in json file
with jsonlines.open("../data/json/pk_pmcs_ner_dec2021/selected_pk_tables.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

random.shuffle(json_list)

trial_set = json_list[0:50]

train_val_set = json_list[50:2050]

test_set = json_list[2050:3050]


#write to file
with jsonlines.open("../data/json/parsed_table_jsons/trial_ner_50.jsonl", mode='w') as writer:
    writer.write_all(trial_set)

#write to file
with jsonlines.open("../data/json/parsed_table_jsons/train_val_ner_2000.jsonl", mode='w') as writer:
    writer.write_all(train_val_set)

#write to file
with jsonlines.open("../data/json/parsed_table_jsons/test_ner_1000.jsonl", mode='w') as writer:
    writer.write_all(test_set)

print("complete")
a=1