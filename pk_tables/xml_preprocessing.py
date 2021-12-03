# imports
import random
from typing import List
import os
from lxml import etree
from io import StringIO
import jsonlines
from tqdm import tqdm


def stringify_children(node):
    """
    Filters and removes possible Nones in texts and tails
    """
    return ''.join(node.itertext())


def tables_to_dict(path_to_file) -> List:
    """
    Converts tables in xml papers to dictionary
    """
    tree = etree.parse(path_to_file)
    # root = tree.getroot()
    article_info = tree.xpath("//front//article-meta")
    tables = tree.xpath("//body//sec//table-wrap")
    # article_title = tree.xpath("//front//article-title")
    # title = [tit.text for tit in article_title]
    for info in article_info:
        try:
            pmc = info.find("article-id[@pub-id-type= 'pmc']").text
            doi = info.find("article-id[@pub-id-type= 'doi']").text
        except:
            print(info)

    table_dicts = []
    for table in tables:
        label = table.find('label').text
        caption = stringify_children(table.find('caption'))
        caption = caption.strip()
        table_xml = etree.tostring(table.find('table'), encoding='unicode')
        html_template = "<!DOCTYPE html><html><body><h4>{0}</h4><head><style> table, th, td {{border: 1px solid black;}}</style></head><body>{1}</body></html>"
        html = html_template.format(caption, table_xml)
        table_dict = {"text": f"PMC{pmc} | {label} | DOI: {doi}",
                      "pmc_link": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc}",
                      "html": html}
        table_dicts.append(table_dict)
        a = 1

    return table_dicts


def write_html(table_dicts, path):
    """Function to convert the table from XML to HTML for visualization during labelling"""
    for i in table_dicts:
        add_front = "<!DOCTYPE html><html><head><style> table, th, td {border: 1px solid black;}</style></head><body>"
        add_end = "</body></html>"
        xx = str(table_dicts[i]["table"])
        xx = add_front + xx[2:]
        xx = xx[:-1] + add_end
        parser = etree.HTMLParser()
        tree_parsed = etree.parse(StringIO(xx), parser)
        tree_parsed.write(path)


def get_file_list(my_dir):
    """return list of all files from my directory"""
    file_list = []
    for filename in os.listdir(my_dir):
        if filename.endswith(".nxml"):
            file = [os.path.join(my_dir, filename).replace("\\", "/")]
            file_list += file
        else:
            continue

    return file_list


def file_list_folders(rootdir):
    file_list = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".nxml"):
                filename = [os.path.join(subdir, file).replace("\\", "/")]
                file_list += filename
    return file_list


def apply_to_all(file_list, json_path):
    """Function to apply tables to dict to all files in file_list"""
    table_list = []
    total_count = 0
    for file in tqdm(file_list):
        try:
            table_dict = tables_to_dict(file)
            if table_dict is not []:
                table_list += table_dict
        except Exception as err:
            pass
            total_count += 1

    with jsonlines.open(json_path, mode='w') as writer:
        writer.write_all(table_list)
    a=1
    print(f"(Total Count= {total_count}")
    return table_list


def split_data(path_to_json, path_to_write):
    """Function to read in json master file and split into trial, train, validation and test sets for labelling"""

    with jsonlines.open(path_to_json) as reader:
        json_list = []
        for obj in reader:
            json_list.append(obj)

    random.Random(4).shuffle(json_list)
    trial = json_list[:50]
    remaining1 = json_list[50:]
    with jsonlines.open(path_to_write + "trial_tableclass.jsonl", mode='w') as writer:
        writer.write_all(trial)

    train = remaining1[:2000]
    remaining2 = remaining1[2000:]
    with jsonlines.open(path_to_write + "train_tableclass.jsonl", mode='w') as writer:
        writer.write_all(train)

    val = remaining2[:500]
    remaining3 = remaining2[500:]
    with jsonlines.open(path_to_write + "val_tableclass.jsonl", mode='w') as writer:
        writer.write_all(val)

    test_tables = remaining3[:1000]
    unused = remaining3[1000:]
    with jsonlines.open(path_to_write + "test_tableclass.jsonl", mode='w') as writer:
        writer.write_all(test_tables)
    with jsonlines.open(path_to_write + "unused_tableclass.jsonl", mode='w') as writer:
        writer.write_all(unused)


def concat_jsonl(in_file_dir: str, out_file_name: str):
    """Concatenate jsonl files, removing duplicates"""
    all_files_list = []
    for filename in os.listdir(in_file_dir):
        with jsonlines.open(os.path.join(in_file_dir, filename)) as reader:
            json_list = []
            for obj1 in reader:
                json_list.append(obj1)
                all_files_list.extend(json_list)

    json_uniques = list({x['text']: x for x in all_files_list}.values())

    print(len(json_uniques))

    with jsonlines.open("../data/json/all_pmcs_ner_dec2021/" + out_file_name, mode='w') as writer:
        writer.write_all(json_uniques)
