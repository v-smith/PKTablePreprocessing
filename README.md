# PKTablePreprocessing

Order of Events: 
1. Parse papers to tables in jsonl file using xml_script.py 
2. Please note that all the PK relevant paper tables (as selected by ferran's doc classifier) are present in pk_pmcs_ner_dec2021/selected_pk_tables.jsonl
3. parse table table into cell data --> scripts/table_data_extraction_script.py
4. parse cells into distilbert to match PK entities --> use_distilbert.py
5. separate relevant tables using distilbert list --> select_relevant_pk_table_forcheck.py
6. transition over to Prodigy_test repo to run vicky-choose-re-tables.py recipe and check over model selections 
7. output prodigy dataset of results and return to this repo
8. run select_relevant_pk_cells.py to subset out relevant cells for annotation based on relevant table selection


Extract Tables from PubMed OA datasets xml files, annotate these with prodigy annotation software and custom recipe.
Get labelled json file from azure and split based on annotator for review session.  
```
python ./scripts/split_annotations.py --azure-file-name table_ner_trial-output.jsonl --save-local False
python ./scripts/InterAnnAgg.py --azure-file-name tableclass-test-params-100-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-trials-1-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-test-covs-100-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-test-params-100-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-test-params-250B-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-test-covs-250B-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-test-params-250A-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-test-covs-250A-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-test-params-200A-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-test-covs-200A-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-test-params-200B-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-test-covs-200B-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-train-params-500-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-train-covs-500-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-train-params-500to750-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-train-covs-500to750-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-train-params-750to1000-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-train-covs-750to1000-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-train-params-1000to1250-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-train-covs-1000to1250-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-train-params-1250to1500-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name tableclass-train-covs-1250to1500-output.jsonl --save-local False
python ./scripts/split_annotations.py --azure-file-name test-corrected-output.jsonl --save-local True
```
Start Review Session
``` 
prodigy review final-covs100 tableclass-test-covs-100-gill,tableclass-test-covs-100-frank,tableclass-test-covs-100-joe,tableclass-test-covs-100-vicky -v choice  
prodigy review final-params100 tableclass-test-params-100-gill,tableclass-test-params-100-frank,tableclass-test-params-100-joe,tableclass-test-params-100-vicky -v choice
prodigy review final-covs250B tableclass-test-covs-250B-vicky,tableclass-test-covs-250B-pum,tableclass-test-covs-250B-frank,tableclass-test-covs-250B-palang -v choice
prodigy review final-params250B tableclass-test-params-250B-vicky,tableclass-test-params-250B-pum,tableclass-test-params-250B-frank,tableclass-test-params-250B-palang -v choice 
prodigy review final-params250A tableclass-test-params-250A-vicky,tableclass-test-params-250A-gill,tableclass-test-params-250A-joe -v choice
prodigy review final-covs250A tableclass-test-covs-250A-vicky,tableclass-test-covs-250A-gill,tableclass-test-covs-250A-joe -v choice   
prodigy review final-params200B tableclass-test-params-200B-vicky,tableclass-test-params-200B-pum,tableclass-test-params-200B-palang,tableclass-test-params-200B-frank -v choice
prodigy review final-covs200B tableclass-test-covs-200B-vicky,tableclass-test-covs-200B-pum,tableclass-test-covs-200B-palang,tableclass-test-covs-200B-frank -v choice   
prodigy review final-params200A tableclass-test-params-200A-vicky,tableclass-test-params-200A-gill,tableclass-test-params-200A-frank,tableclass-test-params-200A-palang -v choice
prodigy review final-covs200A tableclass-test-covs-200A-vicky,tableclass-test-covs-200A-gill,tableclass-test-covs-200A-palang,tableclass-test-covs-200A-frank -v choice 
prodigy review final-params-train200A tableclass-test-params-200A-vicky,tableclass-test-params-200A-gill,tableclass-test-params-200A-frank,tableclass-test-params-200A-palang -v choice
prodigy review final-train-params-500 tableclass-train-params-500-pum,tableclass-train-params-500-palang,tableclass-train-params-500-frank,tableclass-train-params-500-vicky -v choice 
prodigy review final-train-covs-500 tableclass-train-covs-500-other,tableclass-train-covs-500-pum,tableclass-train-covs-500-palang,tableclass-train-covs-500-frank,tableclass-train-covs-500-vicky -v choice
prodigy review final-train-params-500to750 tableclass-train-params-500to750-pum,tableclass-train-params-500to750-vicky,tableclass-train-params-500to750-gill -v choice
prodigy review final-train-covs-500to750 tableclass-train-covs-500to750-vicky,tableclass-train-covs-500to750-palang,tableclass-train-covs-500to750-gill,tableclass-train-covs-500to750-frank -v choice   
prodigy review final-train-covs-750to1000 tableclass-train-covs-750to1000-palang,tableclass-train-covs-750to1000-vicky -v choice
prodigy review final-train-params-750to1000 tableclass-train-params-750to1000-gill,tableclass-train-params-750to1000-frank,tableclass-train-params-750to1000-palang,tableclass-train-params-750to1000-vicky,tableclass-train-params-750to1000-pum -v choice
prodigy review final-train-covs-1000to1250 tableclass-train-covs-1000to1250-frank,tableclass-train-covs-1000to1250-palang,tableclass-train-covs-1000to1250-vicky -v choice
prodigy review final-train-params-1000to1250 tableclass-train-params-1000to1250-palang,tableclass-train-params-1000to1250-frank -v choice
prodigy review final-train-covs-1250-1500 tableclass-train-covs-1250to1500-frank,tableclass-train-covs-1250to1500-vicky,tableclass-train-covs-1250to1500-palang -v choice
prodigy review final-train-params-1250-1500 tableclass-train-params-1250to1500-vicky,tableclass-train-params-1250to1500-frank -v choice
```
Get Final Reviewed Annotations Out
```
python -m prodigy db-out final-covs100 > ./data/final-out//final-test-covs100.jsonl
python -m prodigy db-out final-params100 > ./data/final-out/final-test-params100.jsonl
python -m prodigy db-out final-covs200B > ./data/final-out//final-test-covs200B.jsonl
python -m prodigy db-out final-params200B > ./data/final-out/final-test-params200B.jsonl
python -m prodigy db-out final-covs250B > ./data/final-out//final-test-covs250B.jsonl
python -m prodigy db-out final-params250B > ./data/final-out/final-test-params250B.jsonl
python -m prodigy db-out final-covs250A > ./data/final-out//final-test-covs250A.jsonl
python -m prodigy db-out final-params250A > ./data/final-out/final-test-params250A.jsonl
python -m prodigy db-out final-covs200A > ./data/final-out//final-test-covs200A.jsonl
python -m prodigy db-out final-params200A > ./data/final-out/pars_test/final-test-params200A.jsonl
python -m prodigy db-out final-train-params-500 > ./data/final-out/pars_test/final-train-params-500.jsonl
python -m prodigy db-out final-train-covs-500 > ./data/final-out/covs_test/final-train-covs-500.jsonl
python -m prodigy db-out final-train-params-500to750 > ./data/final-out/pars_test/final-train-params-500to750.jsonl
python -m prodigy db-out final-train-covs-500to750 > ./data/final-out/covs_test/final-train-covs-500to750.jsonl
python -m prodigy db-out final-train-covs-750to1000 > ./data/final-out/covs_test/final-train-covs-750to1000.jsonl
python -m prodigy db-out final-train-params-750to1000 > ./data/final-out/pars_test/final-train-params-750to1000.jsonl
python -m prodigy db-out final-train-covs-1000to1250 > ./data/final-out/covs_test/final-train-covs-1000to1250.jsonl
python -m prodigy db-out final-train-covs-1250-1500 > ./data/final-out/covs_test/final-train-covs-1250-1500.jsonl
python -m prodigy db-out final-train-params-1000to1250 > ./data/final-out/pars_test/final-train-params-1000to1250.jsonl
python -m prodigy db-out final-train-params-1250-1500 > ./data/final-out/pars_test/final-train-params-1250-1500.jsonl
```

Run Grobid Client
```
#do one article 
cd grobid/
./gradlew run #http://localhost:8070/
curl -v --form input=@/home/vsmith/PycharmProjects/PKTablePreprocessing/data/azole_final_papers/PMID1357148.pdf localhost:8070/api/processFulltextDocument

#do a batch 
cd grobid/
./gradlew run #http://localhost:8070/
#use grobid-service, run example.py

```

# Download the files into a single xml file from PubMed FTP site from list of pmc ids or pmids
```
wget https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=13900,13901&retmode=xml&retmax=200 -P /home/vsmith/PycharmProjects/PKTablePreprocessing/data/vicky_pmc_xmls/
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=212403
```

"""
GRPBID ERROR 
Failed error 408: - manual or noncomm 
PMID27324763.pdf  
PMID31971567.pdf
PMID16641468.pdf
PMID25779580.pdf
PMID26239045.pdf
PMID14616415.pdf
PMID17606672.pdf
PMID32468741.pdf
PMID11302829.pdf
"""
