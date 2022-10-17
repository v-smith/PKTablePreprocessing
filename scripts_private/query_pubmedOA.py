from os import listdir
from os.path import isfile, join
from shutil import copyfile

import pandas

only_files = [".".join(f.split(".")[:-1]) for f in listdir("../data/azole_final_papers") if
              isfile(join("../data/azole_final_papers", f))]

pmids = [ids[4:] for ids in only_files]
print("All pmids", len(pmids))
pmids = list(set(pmids))
print("All unique pmids", len(pmids))

# get pmc ids using pmids, using the master file
id_df = pandas.read_csv("../data/PMC-ids.csv", dtype=str)
id_df = id_df[['PMCID', 'PMID']]
pmc_list = id_df["PMCID"].tolist()
pmid_checklist = id_df["PMID"].tolist()

final_list_pmid = []
final_list_pmc = []
for a, b in zip(pmid_checklist, pmc_list):
    if a in pmids:
        final_list_pmid.append(a)
        final_list_pmc.append(b)

# subset those files without pmc ids
pmids_for_grobid = list(set(pmids) - set(final_list_pmid))
print("Final PMC-IDS", len(final_list_pmc))
print("Final PMIDS", len(final_list_pmid))
print("Final PMIDS for Grobid", len(pmids_for_grobid))

# format download command
final_list_pmc_formatted = [x[3:] for x in final_list_pmc]
final_string_pmc_formatted = ",".join(final_list_pmc_formatted)
template = f"""https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={final_string_pmc_formatted}&retmode=xml&retmax=300"""

# write download command out to text file
textfile = open("../data/vicky_pmc_xmls/pmc_download.txt", "w")
textfile.write(template)
textfile.close()
# N.B. To Download the files into a single xml file from PubMed FTP site from list of pmc ids or pmids
'''
wget https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id= -P /home/vsmith/PycharmProjects/PKTablePreprocessing/data/vicky_pmc_xmls/
'''
a = 1

# all other ids, put these pdfs in separate file and pass to grobid
counter = 0
for i in pmids_for_grobid:
    try:
        current_direct = "../data/azole_final_papers/"
        new_direct = "../data/vicky_togrobid_papers/"
        current_file = current_direct + "PMID" + str(i) + ".pdf"
        new_file = new_direct + "PMID" + str(i) + ".pdf"
        copyfile(current_file, new_file)
    except:
        print(i)
        counter += 1

print("Couldn't be copied", counter)

a = 1
