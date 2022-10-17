import pubmed_parser as pp
from tqdm import tqdm
from pk_tables.xml_preprocessing import file_list_folders
from pk_tables.utils import write_jsonl

file_list = file_list_folders('../data/selected_pk_pmc_files')

parsed_pk_papers = []
for xml in tqdm(file_list):
    # get metadata including abstract
    metadata_dict = pp.parse_pubmed_xml(xml)
    full_text_paragraphs = pp.parse_pubmed_paragraph(xml, all_paragraph=True)
    metadata_dict["full_text"] = full_text_paragraphs
    parsed_pk_papers.append(metadata_dict)

print(len(parsed_pk_papers))

# save_results
parsed_pk_papers = write_jsonl("../data/json/pk_pmcs_ner_dec2021/selected_pk_fulltext.jsonl", parsed_pk_papers)
