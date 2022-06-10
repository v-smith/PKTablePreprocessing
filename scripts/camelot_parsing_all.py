#imports
import camelot
import pdfplumber
import pandas as pd
from pk_tables.utils import find_pdf_coords
import os
import cv2
from tqdm import tqdm

#get all image file names
table_images = os.listdir("../data/images/all_pdfs_out")
not_processed = []
#table_images = ["PMID1357148-0001-2_bbox190.50368,664.28705,831.0331,813.3165.jpg"]

#get all image dimensions needed by loading in image
for im_path in tqdm(table_images):
    id_info = os.path.splitext(im_path)[0]
    im_filename = id_info.split("_bbox")[0]
    im = cv2.imread("../data/images/pdf_images/" + im_filename + ".jpg")
    image_dimensions = im.shape  # height, width, channels
    pmid = id_info.split("-")[0]
    page_number = id_info.split("-")[2]
    page_number = page_number.split("_")[0].lstrip("0")
    page_number = int(page_number)
    bbox = id_info.split("_bbox")[1]
    bbox_lst = bbox.split(",")  # x,y,w,h [168.6165, 236.2609, 836.9727, 1572.1078]
    bbox_lst = [float(x) for x in bbox_lst]


    #load in appropriate pdf
    pdf_path = '../data/all_pdf_togrobid/' + str(pmid) + ".pdf"
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number-1]
    bounding_box, bounding_box_camelot = find_pdf_coords(bbox_lst, image_dimensions, page)
    try:
        table = camelot.read_pdf(pdf_path, pages=str(page_number), flavor='stream', table_areas=bounding_box_camelot)
        #camelot.plot(table[0], kind='text').show()
        #camelot.plot(table[0], kind='contour').show()
        df = table[0].df
        csv_filename = "../data/table_csvs/" + pmid + "_" + str(page_number) + ".csv"
        xlsx_filename = "../data/table_xlsx/" + pmid + "_" + str(page_number) + ".xlsx"
        df.to_csv(csv_filename)
        df.to_excel(xlsx_filename)
    except Exception as err:
        not_processed.append(im_path)


print(len(not_processed))
with open("../data/not_processed_camelot.txt", 'w') as output:
    for row in not_processed:
        output.write(str(row) + '\n')
a = 1
