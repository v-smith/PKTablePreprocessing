#imports
import camelot
import pdfplumber
import pandas as pd
import matplotlib


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

    bounding_box = [table_x0, table_y0, table_x1 + 15, table_y1] #(x0, top, x1, bottom)
    bounding_box_camelot= [table_x0, (height - table_y0), (table_x1 + 15), (height-table_y1)]
    bounding_box_camelot = [str(int(x)) for x in bounding_box_camelot]
    bounding_box_camelot = [",".join(bounding_box_camelot)]

    return bounding_box, bounding_box_camelot


with pdfplumber.open('../data/vicky_togrobid_papers/vicky_pmc_togrobid_papers/PMID1605615.pdf') as pdf:
    third_page = pdf.pages[2]
    table_tensor = [168.6165, 236.2609, 836.9727, 1572.1078] #x,y,w,h
    image_dimensions = [2200, 1700, 3] #height, width, channels
    bounding_box, bounding_box_camelot = find_pdf_coords(table_tensor, image_dimensions, third_page)
    third_page = third_page.crop(bounding_box) #(x0, top, x1, bottom)
    im = third_page.to_image(resolution=150)
    im.save("../data/images/cropped.png", format="PNG")
    found_tables = third_page.find_tables(table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
    extracted_tables = third_page.extract_table(
        table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
    #debug_tabs = third_page.debug_tablefinder(
        #table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"})
    df = pd.DataFrame(extracted_tables[1:], columns=extracted_tables[0])

a = 1

tables = camelot.read_pdf('../data/vicky_togrobid_papers/vicky_pmc_togrobid_papers/PMID1605615.pdf', pages='3',
                          flavor='stream', table_areas=bounding_box_camelot) #['168,236,836,1572']
camelot.plot(tables[0], kind='text').show()
camelot.plot(tables[0], kind='contour').show()
df2 = tables[0].df
# tables[0].parsing_report
a = 1
