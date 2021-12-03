# imports
import jsonlines
from typing import List


def replace_list_fromdict(l: list, dictionary: dict) -> List:
    """Function to replace all values in a list with values from dictionary"""
    for i, j in dictionary.items():
        if i in l:
            l[l.index(i)] = j

    return l


def labels_update(covs_list: List[dict]) -> List[dict]:
    """Function to replace accept with new labels in every dic in covariates list"""
    # replacement dict for covs
    label_dict = {1: 1,
                  2: 7,
                  3: 8,
                  4: 9,
                  5: 10}

    not_relevant_dict = {6: 5}

    one_dict = {1: 6}

    for dic in covs_list:
        replace_list_fromdict(dic["accept"], label_dict)
        replace_list_fromdict(dic["accept"], not_relevant_dict)
        replace_list_fromdict(dic["accept"], one_dict)

    return covs_list

def concat_lists(params_list: List[dict], new_covs_list: List[dict]) -> List[dict]:
    """Function to concat 2 lists based on table ID and combine labels for params and covs"""

    for elm1 in params_list:
        for elm2 in new_covs_list:
            if elm2['text'] == elm1['text']:
                for x in elm2["accept"]:
                    if x not in list(elm1["accept"]):
                        elm1['accept'].append(x)
    return params_list


def remove_notrel(json_list: List[dict]):
    for item in json_list:
        if len(item["accept"]) > 1:
            if 5 in item["accept"]:
                item['accept'].remove(5)

    return json_list


def concat_parwithcovs(par_file: str, cov_file: str, out_file: str, check_text: str):
    """Function to concatenate labelling results files from covariate and parameter labelling.
    Please note covariate labels are mapped to new values as per the label_dict"""

    with jsonlines.open(par_file) as reader:
        params_list = []
        for obj in reader:
            params_list.append(obj)

    with jsonlines.open(cov_file) as reader:
        covs_list = []
        for obj in reader:
            covs_list.append(obj)

    new_covs_list = labels_update(covs_list)

    new_json = concat_lists(params_list, new_covs_list)

    final_json = remove_notrel(new_json)

    for i in final_json:
        if i["text"] == check_text:
            print(i["accept"])

    with jsonlines.open("../data/final-out-concat/test/" + out_file, mode='w') as writer:
        writer.write_all(final_json)

    print(f"Length of out list: {len(final_json)}")



