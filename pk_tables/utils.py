import camelot
import pandas as pd
from utils import nerdemo
import multiprocessing
import pathlib
from utils import nerdemo
from transformers import BertTokenizerFast
import jsonlines
import os
from tqdm import tqdm

def find_pdf_coords(table_dims, image_dims, page):
    #pdf page coords
    height = page.height
    width = page.width

    #ration of table to whole image coords
    ratio_x0 = table_dims[0]/image_dims[1]
    ratio_y0 = table_dims[1]/image_dims[0]
    ratio_x1 = table_dims[2]/image_dims[1]
    ratio_y1 = table_dims[3]/image_dims[0]

    #pdf table coords
    table_x0 = width*ratio_x0
    table_y0 = height*ratio_y0
    table_x1 = width*ratio_x1
    table_y1 = height*ratio_y1

    bounding_box = [table_x0, table_y0, table_x1, table_y1] #(x0, top, x1, bottom)
    bounding_box_camelot = [table_x0, (height - table_y0), (table_x1), (height-table_y1)]
    bounding_box_camelot = [str(int(x)) for x in bounding_box_camelot]
    bounding_box_camelot = [",".join(bounding_box_camelot)]

    return bounding_box, bounding_box_camelot


