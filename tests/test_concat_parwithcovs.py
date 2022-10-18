# imports
from pk_tables.concat_parwithcovs import remove_notrel
from pk_tables.concat_parwithcovs import concat_lists, labels_update

covs_list = [{"text": 3, "accept": [3, 5, 6]}, {"text": 1, "accept": [1, 2]}, {"text": 2, "accept": [3, 4]}]
params_list = [{"text": 3, "accept": [2, 5]}, {"text": 1, "accept": [1, 3]}, {"text": 2, "accept": [5, 4]}]

labels_update(covs_list)


def test_concat_lists():
    concat_list = concat_lists(params_list, covs_list)

    assert concat_list == [{'text': 3, 'accept': [2, 5, 8, 10]}, {'text': 1, 'accept': [1, 3, 6, 7]},
                           {'text': 2, 'accept': [5, 4, 8, 9]}]

    remove_nr = remove_notrel(concat_list)

    assert remove_nr == [{'text': 3, 'accept': [2, 8, 10]}, {'text': 1, 'accept': [1, 3, 6, 7]},
                         {'text': 2, 'accept': [4, 8, 9]}]
