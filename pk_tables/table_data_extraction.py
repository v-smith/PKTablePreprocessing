# imports
import pandas as pd
from bs4 import BeautifulSoup
import re
from typing import List, Tuple, Dict, Any, Union
import itertools
import hashlib
import pickle
from IPython.display import HTML, Javascript
import numpy as np


def parse_xml_tables(json_list: List) -> List:
    # convert xml tables to dataframes
    table_dfs_list = convert_htmls_to_dfs(json_list)
    #demo_df = table_dfs_list[0]["table_df"]

    # split any 'modelling' tables with mid-table subheaders (marked with double null rows)
    processed_table_dfs_list = process_table_dfs(table_dfs_list)
    #demo_dfa = processed_table_dfs_list[0]["table_df"]
    #demo_dfb = processed_table_dfs_list[1]["table_df"]

    #parse to value dicts
    parsed_table_dict_list = parse_tables_to_value_dicts(processed_table_dfs_list)
    #demo_dict_list = parsed_table_dict_list[0]["value_dicts"]
    # drop empty value lists
    final_table_dict_list = [x for x in parsed_table_dict_list if x["value_dicts"]]
    jsonl_text_dict_list, hash_list = covert_to_labelling_jsonl(final_table_dict_list)
    write_hashes_html_pickle(hash_list)
    my_pickle = pickle.load(open('../data/json/parsed_table_jsons/table_hashes_2.pkl', 'rb'))
    assert len(my_pickle) == len(hash_list)
    a=1
    return jsonl_text_dict_list


def style_dataframes(df, text, column, row):
    s = df.style.apply(styling_specific_cell, row_idx=row, col_idx=column, text=text,
                       axis=None).set_properties(**{'border': '1.3px solid black', 'color': 'black'})
    styled_html = s.render()
    #file = open("../data/parsed_table_jsons/index.html", "w")
    #file.write(styled_html)
    #file.close()

    return styled_html


def styling_specific_cell(x, row_idx, col_idx, text):
    color = 'background-color: yellow; font-weight: bold'
    font = "font-weight: bold"
    df_styler = pd.DataFrame('', index=x.index, columns=x.columns)
    df_styler.loc[0, :] = font
    #df_styler.loc["level_1"] = font
    df_styler.loc[row_idx, col_idx] = color
    a=1
    return df_styler


def hash_html_tables(html, text):
    encoded_html = html.encode("utf8")
    h = hashlib.sha3_256()
    h.update(encoded_html)
    hashed_html = h.hexdigest()
    return {"hash": hashed_html, "html": html, "text": text}


def write_hashes_html_pickle(hash_list):
    hash_dict = {}
    for item in hash_list:
        hash = item["hash"]
        html = item["html"]
        hash_dict[hash] = html

    with open(r"../data/json/parsed_table_jsons/table_hashes_2.pkl", "wb") as f:
        pickle.dump(hash_dict, f)

    #my_pickle = pickle.load(open('../data/parsed_table_jsons/table_hashes_2.pkl', 'rb'))
    #assert len(my_pickle) == len(hash_list)
    a=1


def df_style(val):
    # bolding a value???
    return 'font-weight: bold'


def convert_to_set(name: str, list_of_dicts: List[Dict]) -> List[Dict]:
    a = 1
    row_non = [y[name] for y in list_of_dicts]
    row_non = [item for sublist in row_non for item in sublist]
    row_non = [dict(t) for t in {tuple(d.items()) for d in row_non}]
    return row_non


def covert_to_labelling_jsonl(final_table_dict_list: List[Dict]) -> List[Dict]:
    all_tables_for_jsonl = []
    hash_list = []
    for item in final_table_dict_list:
        table_df = item["table_df"]
        id = item["identifier"]
        dics = item["value_dicts"]
        # get unique non_numeric cell values
        all_non_vals = convert_to_set('non_numeric_values', dics)
        # drop if "Unnamed"
        all_non_vals_cleaned = [x for x in all_non_vals if not re.match("Unnamed", x["value"])]
        text_all_vals = [x["value"] for x in all_non_vals_cleaned]
        text_label_col = [x["column"] for x in all_non_vals_cleaned]
        text_label_row = [x["row"] for x in all_non_vals_cleaned]

        table_for_jsonl = []
        #sub_hash_list = []
        for t, c, r in zip(text_all_vals, text_label_col, text_label_row):
            styled_html = style_dataframes(table_df, t, c, r)
            hash_dict = hash_html_tables(styled_html, t)
            hash_list.append(hash_dict)
            table_for_jsonl.append({"text": t, "col": c, "row": r, "html": hash_dict["hash"], "table_id": id,
                                    "meta": {"DOI": id, "Table": id}}),

        all_tables_for_jsonl.extend(table_for_jsonl)
        sorted_all_tables_for_jsonl = sorted(all_tables_for_jsonl, key=lambda d: (d['col'], d["row"]))

        a = 1
    return sorted_all_tables_for_jsonl, hash_list


def parse_tables_to_value_dicts(processed_table_dfs_list: List) -> List:
    final_parsed_list = []
    for item in processed_table_dfs_list:
        # table_html = html["html"]
        table_df = item["table_df"]
        try:
            table_df = table_df.T.reset_index(drop=False).T
        except:
            print(item["identifier"])
        updated_table_df = table_df.reset_index(drop=True)
        table_dict = updated_table_df.to_dict()
        caption = item["caption"]
        identifier = item["identifier"]
        link = item["link"]
        # separate out headers rows and columns
        columns, row_headers = separate_columns(table_dict)
        processed_row_headers = process_header_cols(row_headers)
        numeric_values, non_numeric_values = determine_numeric_cells(columns, processed_row_headers)
        value_dicts = get_value_dict(numeric_values, non_numeric_values)
        final_parsed_list.append({"value_dicts": value_dicts, "identifier": identifier, "link": link, "caption": caption, "table_df": updated_table_df})
    a=1
    return final_parsed_list


def get_value_dict(numeric_values: List[Dict], non_numeric_values: List[Dict]) -> \
        List[Dict]:
    # bring together each numeric value with all its relevant headers
    value_dicts = []
    for value in numeric_values:
        col_id = value["column"]
        row_id = value["row"]
        relevant_non_numeric = [x for x in non_numeric_values if x["column"] == col_id or x["row"] == row_id]
        value_dicts.append(
            {"numeric_value": value, "non_numeric_values": relevant_non_numeric})
    return value_dicts


def determine_numeric_cells(columns: List, header_rs: List) -> Tuple:
    numeric_values = []
    non_numeric_values = []
    nans_list = []
    counter = 0
    for dic in columns:
        counter += 1
        for k, v in dic.items():
            if match(v):
                numeric_values.append({"row": k, "column": (counter - 1), "value": v, "numeric": True})
            elif pd.isna(v):
                nans_list.append(({"row": k, "column": (counter - 1), "value": v}))
            else:
                non_numeric_values.append({"row": k, "column": (counter - 1), "value": v, "numeric": False})
    a=1
    # check for roll up rows 
    nans_list, roll_up_rows = check_rollup_rows(nans_list, header_rs, counter)
    if roll_up_rows:
        non_numeric_values.extend(roll_up_rows)

    # check remaining nans for numeric-ness 
    numeric_values, non_numeric_values = process_nan_values(nans_list, numeric_values, non_numeric_values)
    return numeric_values, non_numeric_values


def check_rollup_rows(nans_list: List, row_headers: List, counter: int) -> Tuple[
    Union[list, list], List[Dict[str, Union[bool, Any]]]]:
    """Checks for all nan rows except for header rows and if so rolls header out along the row"""
    number_of_cols = counter - 1
    roll_up_rows = []
    for row_num in row_headers:
        row_header_value = row_num["value"]
        row = row_num["row"]
        nans_per_row = [x for x in nans_list if x["row"] == row]
        if len(nans_per_row) == number_of_cols:
            for row_nan in nans_per_row:
                roll_up_rows.append(
                    {"row": row_nan["row"], "column": row_nan["column"], "value": row_header_value, "numeric": False})
                nans_list = [x for x in nans_list if x["row"] != row_nan["row"] and x["column"] != row_nan["column"]]
            a = 1
    return nans_list, roll_up_rows


def process_nan_values(nans_list: List[Dict], numeric_values: List[Dict], non_numeric_values: List[Dict]):
    """Assumptions: if all other values in a row or column are numeric (excluding header columns and rows)
     then this function interprets a nan as numeric and adds it to the numeric values list """
    for nan in nans_list:
        row = nan["row"]
        col = nan["column"]
        value = nan["value"]
        matching_numeric_rows = [x for x in numeric_values if x["row"] == row]
        matching_numeric_cols = [x for x in numeric_values if x["column"] == col]
        if matching_numeric_cols or matching_numeric_rows:
            numeric_values.append({"row": row, "column": col, "value": value, "numeric": True})
        else:
            non_numeric_values.append({"row": row, "column": col, "value": value, "numeric": False})
    return numeric_values, non_numeric_values


def process_header_rows(row_headers: List) -> List[Dict]:
    processed_row_headers = []
    counter = 0
    for header in row_headers:
        counter += 1
        if type(header) == tuple:
            for x in header:
                processed_row_headers.append({"row": "header", "column": counter, "value": x, "numeric": False})
        else:
            processed_row_headers.append({"row": "header", "column": counter, "value": header, "numeric": False})
    return processed_row_headers


def process_header_cols(col_headers: List) -> List[Dict]:
    processed_col_headers = []
    for header in col_headers:
        for k, v in header.items():
            processed_col_headers.append({"row": k, "column": 0, "value": v, "numeric": False})
    return processed_col_headers


def separate_columns(table_dict: Dict):
    columns = []
    row_headers = []
    counter = 0
    for k, v in table_dict.items():
        counter += 1
        column = v
        columns.append(column)
        if counter == 1:
            row_headers.append(column)
    return columns, row_headers


def match(s: str) -> bool:
    # ([^a-zA-Z]+)$, (\d+(?:\.\d+)?)
    if s == "nan":
        return True
    elif len(re.findall("([a-zA-Z]+)", str(s))) >= 1:
        return False
    else:
        return True


def process_table_dfs(table_dfs_list: List) -> List[Dict]:
    processed_table_dfs_list = []
    for entry in table_dfs_list:
        my_df = entry["table_df"]
        caption = entry["caption"]
        ident = entry["identifier"]
        link = entry["link"]
        # index_of_null_rows = my_df[my_df.isnull().all(axis=1)].index
        #TODO: WHAT DOES THIS DO!?!?!?
        #consecutive_list1 = [as_range(g) for _, g in
                            #itertools.groupby([9, 10, 12], key=lambda n, c=itertools.count(): n - next(c))]
        consecutive_list1 = [as_range(p) for _, p in
                             itertools.groupby([13,14,15], key=lambda n, c=itertools.count(): n - next(c))]
        consecutive_list = [x for x in consecutive_list1 if x is not None]
        if len(consecutive_list) == 1:
            df1, df1_header, df2, df2_header = split_on_subheader_nulls(ident, my_df, consecutive_list)
            if df2_header != "1":
                processed_table_dfs_list.append({"table_df": df1, "caption": caption, "identifier": df1_header, "link": link})
                processed_table_dfs_list.append({"table_df": df2, "caption": caption, "identifier": df2_header, "link": link})
            else:
                processed_table_dfs_list.append({"table_df": df1, "caption": caption, "identifier": df1_header, "link": link})
        else:
            processed_table_dfs_list.append(entry)

    return processed_table_dfs_list


def split_on_subheader_nulls(ident: str, my_df: pd.DataFrame, consecutive_list: List):
    consecutive_list = [item for sublist in consecutive_list for item in sublist]
    df1 = my_df[:consecutive_list[0]]
    df2 = my_df[consecutive_list[0]:]
    df1 = tidy_up_null_rows_df(df1)
    df2 = tidy_up_null_rows_df(df2)
    if not df2.empty:
        new_header = df2.iloc[0]
        df2 = df2[1:]
        df2.columns = new_header
        df2 = df2.reset_index(drop=True)
        df2_label = f"{ident} SPLIT-B"
        df1_label = f"{ident} SPLIT-A"
    else:
        df1_label = ident
        df2 = 1
        df2_label = "1"

    return df1, df1_label, df2, df2_label


def tidy_up_null_rows_df(my_df: pd.DataFrame) -> pd.DataFrame:
    tidy_df = my_df.dropna(axis=0, how='all')
    return tidy_df


def as_range(iterable):
    element = list(iterable)
    if len(element) > 1:
        return [element[0], element[-1]]
    else:
        pass


def convert_htmls_to_dfs(json_list: List) -> List:
    df_list = []
    counter = 0
    for item in json_list:
        html = item["html"]
        text = item["text"]
        link = item["pmc_link"]
        caption = re.findall("\<h4>(.*?)\</h4>", html)
        try:
            pd_table = pd.read_html(html, header=None)  # header=[0]
            table_df = pd_table[0]
            df_list.append({"table_df": table_df, "caption": caption, "identifier": text, "link": link})
        except ValueError:
            counter += 1
        a=1
    print(f"Could not convert {counter} htmls to dataframes")
    return df_list


def parse_html_table(html: str) -> pd.DataFrame:
    """
    Converts html table into structured dataframe with columns and rows using beautiful soup package
    """
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find_all('table')[0]

    n_columns = 0
    n_rows = 0
    column_names = []
    row_marker = 0

    for row in table.find_all('tr'):

        # Determine the number of rows in the table
        td_tags = row.find_all('td')
        if len(td_tags) > 0:
            n_rows += 1
            if n_columns == 0:
                # Set the number of columns for our table
                n_columns = len(td_tags)

        # Handle column names if we find them
        th_tags = row.find_all('th')
        if len(th_tags) > 0 and len(column_names) == 0:
            for th in th_tags:
                column_names.append(th.get_text())

    # Safeguard on Column Titles
    if len(column_names) > 0 and len(column_names) != n_columns:
        raise Exception("Column titles do not match the number of columns")

    columns = column_names if len(column_names) > 0 else range(0, n_columns)
    df = pd.DataFrame(columns=columns,
                      index=range(0, n_rows))
    row_marker = 0
    for row in table.find_all('tr'):
        column_marker = 0
        columns = row.find_all('td')
        for column in columns:
            df.iat[row_marker, column_marker] = column.get_text()
            column_marker += 1
        if len(columns) > 0:
            row_marker += 1

    # Convert to float if possible
    for col in df:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass

    return df