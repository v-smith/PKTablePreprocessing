from pk_tables.xml_preprocessing_azoles import apply_to_all, file_list_folders
import os
from shutil import copyfile

final_list_pmid = ['9869578', '11181397', '11302829', '9527794', '9661037', '12121931', '12121932', '8878612', '9371367', '9420044', '2848442', '2847635', '12936975', '8388198', '1605615', '14982768', '15155217', '15273137', '9378812', '15328123', '15388441', '8527290', '16436724', '16641468', '16723557', '16940139', '16982783', '17074798', '17088483', '17116670', '17101682', '17210771', '17145785', '9764970', '9690949', '14616411', '14616414', '14616412', '14616408', '14616409', '14616407', '14616415', '14616416', '14748822', '17517842', '11422002', '17606672', '17646413', '17073891', '17495874', '25084200', '25199779', '25512407', '25451051', '25645660', '25886578', '25779580', '25801557', '25824210', '25999694', '26149987', '26259790', '12562722', '27021324', '27121401', '27324763', '26239045', '27367040', '27636722', '28604474', '28370390', '28848009', '29038273', '29581122', '29607533', '29712663', '29315506', '29975796', '30297369', '29679234', '30744151', '30913226', '30670416', '31114213', '31171022', '26612870', '31971567', '31696544', '32468741', '32457106', '32899425', '31768008', '32847935', '32988816', '34031053', '34097481', '34152812', '34734029', '34370587', '34683408', '34721012', '34458906', '34655050']
pmc_ids_needed = ['PMC89033', 'PMC90410', 'PMC90507', 'PMC105422', 'PMC105699', 'PMC127341', 'PMC127364', 'PMC163504', 'PMC164162', 'PMC164194', 'PMC175857', 'PMC175928', 'PMC182636', 'PMC187759', 'PMC188462', 'PMC353067', 'PMC415618', 'PMC478538', 'PMC499974', 'PMC514780', 'PMC521869', 'PMC1365108', 'PMC1366875', 'PMC1472190', 'PMC1479147', 'PMC1563530', 'PMC1693991', 'PMC1797667', 'PMC1797701', 'PMC1797737', 'PMC1797752', 'PMC1803109', 'PMC1803130', 'PMC1873673', 'PMC1873980', 'PMC1884310', 'PMC1884311', 'PMC1884312', 'PMC1884314', 'PMC1884315', 'PMC1884316', 'PMC1884317', 'PMC1884318', 'PMC1884431', 'PMC1932535', 'PMC2014476', 'PMC2043216', 'PMC2043278', 'PMC2203246', 'PMC3488349', 'PMC4155516', 'PMC4249414', 'PMC4335848', 'PMC4335884', 'PMC4338618', 'PMC4403850', 'PMC4432122', 'PMC4432127', 'PMC4432131', 'PMC4435089', 'PMC4538523', 'PMC4576128', 'PMC4687480', 'PMC4879430', 'PMC4954923', 'PMC4997866', 'PMC5057355', 'PMC5061797', 'PMC5345593', 'PMC5538305', 'PMC5555860', 'PMC5655095', 'PMC5740334', 'PMC5971586', 'PMC6005582', 'PMC6021660', 'PMC6037619', 'PMC6177717', 'PMC6256753', 'PMC6326087', 'PMC6406770', 'PMC6435162', 'PMC6496160', 'PMC6497849', 'PMC6554926', 'PMC6937013', 'PMC7069473', 'PMC7098863', 'PMC7335652', 'PMC7526808', 'PMC7557832', 'PMC7643732', 'PMC7669726', 'PMC7674053', 'PMC8284459', 'PMC8370213', 'PMC8370249', 'PMC8506700', 'PMC8522747', 'PMC8538714', 'PMC8548711', 'PMC8598294', 'PMC8602551']

file_list = file_list_folders("../data/xml/")

remove_list = []
papers_for_processing = []
for file in file_list:
    pmc_id = os.path.basename(file)
    pmc_id2 = os.path.splitext(pmc_id)[0]
    if pmc_id2 in pmc_ids_needed:
        papers_for_processing.append(file)
        remove_list.append(pmc_id2)

ids_to_pass_through_grobid= [y for x, y in zip(pmc_ids_needed, final_list_pmid) if x not in remove_list]

counter = 0
for i in ids_to_pass_through_grobid:
    counter += 1
    current_direct = "../data/azole_final_papers/"
    new_direct = "../data/vicky_togrobid_papers/vicky_pmc_togrobid_papers/"
    current_file = current_direct + "PMID" + str(i) + ".pdf"
    new_file = new_direct + "PMID" + str(i) + ".pdf"
    copyfile(current_file, new_file)
print(counter)

table_list = apply_to_all(papers_for_processing, "../data/json/azoles/pmcs_parsed/test_pmcazole.jsonl")

a=1
'''
requests= []
for i in pmc_ids_needed:
    request = f"""https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={i}&retmode=xml"""
    requests.append(request)
# write download command out to text file
with open("../data/vicky_pmc_xmls/pmc_download.txt", "wt") as textfile:
    textfile.write("\n".join(requests))
textfile.close()


# all_tables = multiarticle_tables_to_dict('../data/vicky_pmc_xmls/efetch_azolefinal.xml', "../data/json/azoles/pmcs_parsed/test_pmcazole.jsonl")
file = str('../data/vicky_pmc_xmls/efetch_azolefinal.xml')
tree = etree.parse(file)
articles = tree.xpath("//pmc-articleset//article")

tables_test_list = []
tables_test = tree.xpath("//pmc-articleset//article//body//sec//table-wrap")
for table in tables_test:
    try:
        label = table.find('label').text
        caption = stringify_children(table.find('caption'))
        caption = caption.strip()
        table_xml = etree.tostring(table.find('table'), encoding='unicode')
        html_template = "<!DOCTYPE html><html><body><h4>{0}</h4><head><style> table, th, td {{border: 1px solid black;}}</style></head><body>{1}</body></html>"
        html = html_template.format(caption, table_xml)
        table_dict = {"html": html, "caption": caption}
        tables_test_list.append(table_dict)
    except Exception as err:
        pass

tables_uniques = list({v['caption']:v for v in tables_test_list}.values())

article_infos = []
table_list_2 = []
error_list= []
for article in articles:
    article_info = article.find(".//front//article-meta")
    try:
        pmc = article_info.find("article-id[@pub-id-type='pmc']").text
        pmid = article_info.find("article-id[@pub-id-type='pmid']").text
        article_infos.append(pmc)
    except Exception as err:
        print(f"No pmc or doi: {article_info}")

    table_dicts_2 = []
    tables = article.findall(".//body//sec//table-wrap")
    for table in tables:
        if table.find('table') is not None:
            label = table.find('label').text
            caption = stringify_children(table.find('caption'))
            caption = caption.strip()
            table_xml = etree.tostring(table.find('table'), encoding='unicode')
            html_template = "<!DOCTYPE html><html><body><h4>{0}</h4><head><style> table, th, td {{border: 1px solid black;}}</style></head><body>{1}</body></html>"
            html = html_template.format(caption, table_xml)
            table_dict = {"text": f"PMC{pmc} | {label} | PMID: {pmid}",
                          "pmc_link": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc}",
                          "html": html,
                          "caption": caption}
            table_dicts_2.append(table_dict)
        else:
            error_list.append(pmc)

    table_list_2.extend(table_dicts_2)

print(list(set(error_list)))
print(len(list(set(error_list))))
table_list_2 = list({v['text']:v for v in table_list_2}.values())
a = 1

with jsonlines.open("../data/json/azoles/pmcs_parsed/test_pmcazole.jsonl", mode='w') as writer:
    writer.write_all(table_list_2)

a = 1
'''
