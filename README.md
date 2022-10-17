# PKTablePreprocessing

This project provides scripts to parse, preprocess, annotate and review Pharmacokinetic tables and table cell text from the PubMed Open Access Subset of papers in XML formt. 

## Table Preprocessing Steps: 
1a. Untar pubmed files
1b. Parse papers to tables in jsonl file using xml_script.py 
2. Please note that all the PK relevant paper tables are present in data/pk_pmcs_ner_dec2021/selected_pk_tables.jsonl
3. parse tables into cell data --> scripts/table_data_extraction_script.py
4. parse cells into distilbert to match PK entities --> use_distilbert.py
5. separate relevant tables using distilbert list --> select_relevant_pk_table_forcheck.py
6. transition over to Prodigy_test repo to run vicky-choose-re-tables.py recipe and check over model selections 
7. output prodigy dataset of results and return to this repo
8. run select_relevant_pk_cells.py to subset out relevant cells for annotation based on relevant table selection


Extract Tables from PubMed OA datasets xml files, annotate these with prodigy annotation software.
```angular2html
prodigy table-ner
```
Get labelled json file from azure and split based on annotator for review session.  
```
python ./scripts/split_annotations.py --azure-file-name table_ner_trial-output.jsonl --save-local False
```
Start Review Session
``` 
prodigy review final-covs100 tableclass-test-covs-100-gill,tableclass-test-covs-100-frank,tableclass-test-covs-100-joe,tableclass-test-covs-100-vicky -v choice  
```
Get Final Reviewed Annotations Out
```
python -m prodigy db-out final-covs100 > ./data/final-out//final-test-covs100.jsonl
```

## Grobid Client
```
#one article 
cd grobid/
./gradlew run #http://localhost:8070/
curl -v --form input=@/home/vsmith/PycharmProjects/PKTablePreprocessing/data/azole_final_papers/PMID1357148.pdf localhost:8070/api/processFulltextDocument

#batch 
cd grobid/
./gradlew run #http://localhost:8070/
#use grobid-service, run example.py
```

## Download the files into a single xml file from PubMed FTP site from list of pmc ids or pmids
```
wget https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=13900,13901&retmode=xml&retmax=200 -P /home/vsmith/PycharmProjects/PKTablePreprocessing/data/vicky_pmc_xmls/
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=212403
```
