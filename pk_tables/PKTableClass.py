from lxml import etree
import pandas as pd
from io import StringIO
import os
import re
import numpy as np


class PKTable(object):

    def __init__(self):
        self.


    # take unlabelled table data and convert to format to label with ner model
    def convert_html_to_df(self):
        self.


    def parse_to_value_dict(self):


    def convert_to_ner_jsonl(self):



    # apply the ner model and get back non-numeric cells
    def apply_ner_model(self):



    # apply entity linker
    def apply_entity_linker(self):


    # apply relations extraction (rule-based)
    def apply_relation_mapper(self):



    # apply postprocessing and into DB
    def post_process(self):
