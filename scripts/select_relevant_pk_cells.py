import jsonlines
from operator import itemgetter
from itertools import groupby
import collections

with jsonlines.open("../data/json/cell_entities/parsed_test_ner_1000_entities.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

with jsonlines.open(
        "../data/json/pk_tablesclass_data/relevant/prodigy-out/table_choose_test_1000_final.jsonl") as reader2:
    json_list2 = []
    for obj2 in reader2:
        json_list2.append(obj2)

not_rel_list = []
rem_list = []
rel_list = []
for item in json_list2:
    if item["accept"] == [1]:
        not_rel_list.append(item["table_id"])
    elif not item["accept"]:
        rel_list.append(item["table_id"])
    else:
        rem_list.append(item["table_id"])

print((len(not_rel_list) + len(rem_list) + len(rel_list)))
print(len(json_list2))

keep_rel_list = []
keep_not_rel_list = []
for cell in json_list:
    if cell["table_id"] in rel_list:
        keep_rel_list.append(cell)
    elif cell["table_id"] in not_rel_list:
        keep_not_rel_list.append(cell)

# split into list of lists of dicts
result = collections.defaultdict(list)
for d in keep_rel_list:
    result[d['table_id']].append(d)
keep_rel_result_list = list(result.values())

keep_rel_no_repeats = []
for lst in keep_rel_result_list:
    texts = []
    for dic in lst:
        if dic["text"] not in texts:
            keep_rel_no_repeats.append(dic)
            texts.append(dic["text"])

print(len(keep_rel_list))
print(len(keep_not_rel_list))
print((len(json_list)))

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

rel_list_split = list(split(rel_list, 6))

def split_diclsts(diclsts, split_list):
    new_lists = []
    for split in split_list:
        new_list = [x for x in diclsts if x["table_id"] in split]
        new_lists.append(new_list)
    return new_lists

split_keep_rel_no_repeats = split_diclsts(keep_rel_no_repeats, rel_list_split)
print([len(x) for x in split_keep_rel_no_repeats])
a = 1

#write out split list for labelling with no repeats
counter = 0
for lines in split_keep_rel_no_repeats:
    counter += 1
    with jsonlines.open(("../data/json/cell_entities/pk_relevant/split_no_repeats/" + "parsed_test_ner_1000_entities_Rel_NoReps_" + str(counter) + ".jsonl"),
                        mode='w') as writer:
        writer.write_all(lines)

#write out full relevant list for this dataset all cells (inc repeats)
with jsonlines.open(("../data/json/cell_entities/pk_relevant/" + "parsed_test_ner_1000_entities_Rel.jsonl"),
                    mode='w') as writer:
    writer.write_all(keep_rel_list)

#write out full not relevant list for this dataset with all cells (inc repeats)
with jsonlines.open("../data/json/cell_entities/not_relevant/" + "parsed_test_ner_1000_entities_NotRel.jsonl",
                    mode='w') as writer:
    writer.write_all(keep_not_rel_list)

a = 1
