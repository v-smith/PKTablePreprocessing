import multiprocessing
import pathlib
from utils import nerdemo
from transformers import BertTokenizerFast
import jsonlines
import os
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "false"
###############################################################################

#import my table data and process to input into the model
with jsonlines.open("../data/json/parsed_pk_pmcs_ner_dec2021/parsed_val_ner_.jsonl") as reader:
    json_list = []
    for obj1 in reader:
        json_list.append(obj1)

#json_list = json_list[:10]

############################ Example Data #####################################

cells = ["Vd", "Cmax", "t(1/2)", "area under the curve", "tmax", "hello", "no", "yes", "tdog"]

############################ Predict Entities ################################
PATH = pathlib.Path(__file__).parent
# NER_DATA_PATH = PATH.joinpath("../datasets/pknerdemo").resolve()
NER_MPATH = "../models/distilbert-epoch=0012-val_f1_strict=0.85.ckpt"
GPUS = False  # torch.cuda.is_available()
CPUS = multiprocessing.cpu_count()
NER_MODEL = nerdemo.load_pretrained_model(model_checkpoint_path=NER_MPATH, gpu=GPUS)
NER_TOKENIZER = BertTokenizerFast.from_pretrained(NER_MODEL.bert.name_or_path)

def predic_ner(inp_text: str):
    spacy_doc = nerdemo.SPACY_MODEL(inp_text)
    sentences = []
    sentences_offsets = []
    for s in spacy_doc.sents:
        sentences.append(s.text)
        sentences_offsets.append((s.start_char, s.end_char))

    predicted_entities = nerdemo.predict_pl_bert_ner(inp_texts=sentences, inp_model=NER_MODEL,
                                                     inp_tokenizer=NER_TOKENIZER,
                                                     batch_size=8,
                                                     n_workers=CPUS)

    final_ent_offsets = []
    for sent_ents, sent_offs in zip(predicted_entities, sentences_offsets):
        sent_ents = nerdemo.clean_instance_span(sent_ents)
        for tmp_ent in sent_ents:
            tmp_ent['start'] = tmp_ent['start'] + sent_offs[0]
            tmp_ent['end'] = tmp_ent['end'] + sent_offs[0]
            final_ent_offsets.append(tmp_ent)

    return final_ent_offsets

##############################
instance_list = []
for item in tqdm(json_list):
    out_ents = predic_ner(inp_text=item["text"])
    instance = [dict(text=item["text"], ents=out_ents, col=item["col"], row=item["row"], table_id=item["table_id"], html=item["html"], meta=item["meta"])]
    instance_list.append(instance)

instance_list = [item for sublist in instance_list for item in sublist]

with jsonlines.open("../data/json/cell_entities/" + "parsed_val_ner__entities.jsonl", mode='w') as writer:
    writer.write_all(instance_list)

#print(*instance_list, sep="\n")

pk_ent_tableid_list = []
for inst in instance_list:
    if inst["ents"]:
        pk_ent_tableid_list.append(inst["table_id"])

pk_ent_tableid_list_uniques = list(set(pk_ent_tableid_list))

with open("../data/json/relevant_ids/" + "parsed_val_ner__ids_distilbert.txt", "w") as f:
    f.write("\n".join(pk_ent_tableid_list_uniques))


a = 1
