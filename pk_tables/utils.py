from pathlib import Path
import ujson


def write_jsonl(file_path, lines):
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))


def read_jsonl(file_path):
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open(encoding='utf-8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


def find_pdf_coords(table_dims, image_dims, page):
    """Get coordinates of a .pdf page on an image"""
    # pdf page coords
    height = page.height
    width = page.width

    # ration of table to whole image coords
    ratio_x0 = table_dims[0] / image_dims[1]
    ratio_y0 = table_dims[1] / image_dims[0]
    ratio_x1 = table_dims[2] / image_dims[1]
    ratio_y1 = table_dims[3] / image_dims[0]

    # pdf table coords
    table_x0 = width * ratio_x0
    table_y0 = height * ratio_y0
    table_x1 = width * ratio_x1
    table_y1 = height * ratio_y1

    bounding_box = [table_x0, table_y0, table_x1, table_y1]  # (x0, top, x1, bottom)
    bounding_box_camelot = [table_x0, (height - table_y0), (table_x1), (height - table_y1)]
    bounding_box_camelot = [str(int(x)) for x in bounding_box_camelot]
    bounding_box_camelot = [",".join(bounding_box_camelot)]

    return bounding_box, bounding_box_camelot
