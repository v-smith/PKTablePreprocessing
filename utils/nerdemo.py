from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torch
from nervaluate import Evaluator
from transformers import AutoModel
from transformers.models.bert.modeling_bert import BertModel
import spacy

SPACY_MODEL = spacy.load("en_core_sci_sm", disable=["ner"])


def predict_pl_bert_ner(inp_texts, inp_model, inp_tokenizer, batch_size, n_workers):
    encodings = inp_tokenizer(inp_texts, padding=True, truncation=True, max_length=inp_model.seq_len,
                              return_offsets_mapping=True, return_overflowing_tokens=True)
    predict_dataset = PKDatasetInference(encodings=encodings)
    predict_loader = DataLoader(predict_dataset, batch_size=batch_size, num_workers=n_workers)
    inp_model.eval()
    predicted_entities = []
    overflow_to_sample = []
    all_seq_end = []

    for idx, batch in tqdm(enumerate(predict_loader)):
        with torch.no_grad():
            batch_logits = inp_model(input_ids=batch['input_ids'],
                                     attention_masks=batch['attention_mask']).to('cpu')

            batch_predicted_entities = predict_bio_tags(model_logits=batch_logits, inp_batch=batch,
                                                        id2tag=inp_model.id2tag)

        for seq_end, omap in zip(batch['offset_mapping'], batch['overflow_to_sample_mapping']):
            all_seq_end.append(seq_end.flatten().max().item())
            overflow_to_sample.append(omap.item())

        predicted_entities += batch_predicted_entities

    predicted_entities = remap_overflowing_entities(predicted_tags=predicted_entities, all_seq_end=all_seq_end,
                                                    overflow_to_sample=overflow_to_sample, original_texts=inp_texts,
                                                    offset_mappings=encodings["offset_mapping"]
                                                    )
    return predicted_entities


def remap_overflowing_entities(predicted_tags: List[List[str]], all_seq_end: List[int], overflow_to_sample: List[int],
                               original_texts: List[str], offset_mappings: List[List[Tuple[int, int]]]) -> List[
    List[Dict]
]:
    if len(set(overflow_to_sample)) == len(overflow_to_sample):  # case with no overflowing tokens
        assert len(set(overflow_to_sample)) == len(original_texts)
        tags_per_sentence = predicted_tags
        offset_mappings_rearranged = offset_mappings
    else:
        # Case in which we have overflowing indices
        assert len(all_seq_end) == len(predicted_tags)
        print("Remapping Overflowing")

        all_o_to_s = []
        tags_per_sentence = []
        offset_mappings_rearranged = []
        for i, (ents, send, o_to_s, offsets) in enumerate(zip(predicted_tags, all_seq_end, overflow_to_sample,
                                                              offset_mappings)):
            if o_to_s not in all_o_to_s:
                # tags original sentence
                all_o_to_s.append(o_to_s)
                offset_mappings_rearranged.append(offsets)
                tags_per_sentence.append(ents)
            else:
                # overflowing tags
                new_offsets = offset_mappings_rearranged[-1] + offsets
                new_entities = tags_per_sentence[-1] + ents

                tags_per_sentence = tags_per_sentence[:-1]  # remove last element
                offset_mappings_rearranged = offset_mappings_rearranged[:-1]

                offset_mappings_rearranged.append(new_offsets)
                tags_per_sentence.append(new_entities)  # re-append last element + new one

            assert len(all_o_to_s) == len(set(all_o_to_s))

    entity_tokens = [bio_to_entity_tokens(tag_prediction) for tag_prediction in tags_per_sentence]

    assert len(tags_per_sentence) == len(original_texts) == len(entity_tokens) == len(offset_mappings_rearranged)

    outputs = []
    for offsets, entities, tag in zip(offset_mappings_rearranged, entity_tokens, tags_per_sentence):
        tmp_outputs = []
        for entity in entities:
            entity["start"] = int(offsets[entity["token_start"]][0])
            entity["end"] = int(offsets[entity["token_end"]][1])
            entity["tags"] = tag
            tmp_outputs.append(entity)
        outputs.append(tmp_outputs)

    return outputs


class PKDatasetInference(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["sentence_idx"] = idx
        return item

    def __len__(self):
        return len(self.encodings.encodings)


def predict_bio_tags(model_logits: torch.Tensor, inp_batch: Dict[str, torch.Tensor],
                     id2tag: Dict[int, str]):
    predictions = model_logits.argmax(dim=2)

    tag_predictions = [[id2tag[prediction.item()] for mask, prediction in zip(att_masks, id_preds) if mask.item() == 1]
                       for att_masks, id_preds in zip(inp_batch["attention_mask"], predictions)]

    return tag_predictions


def bio_to_entity_tokens(inp_bio_seq: List[str]) -> List[Dict]:
    """
    Gets as an input a list of BIO tokens and returns the starting and end tokens of each span
    @return: The return should be a list of dictionary spans in the form of [{"token_start": x,"token_end":y,"label":""]
    """
    out_spans = []

    b_toks = sorted([i for i, t in enumerate(inp_bio_seq) if "B-" in t])  # Get the indexes of B tokens
    sequence_len = len(inp_bio_seq)
    for start_ent_tok_idx in b_toks:
        entity_type = inp_bio_seq[start_ent_tok_idx].split("-")[1]
        end_ent_tok_idx = start_ent_tok_idx + 1
        if start_ent_tok_idx + 1 < sequence_len:  # if it's not the last element in the sequence
            for next_token in inp_bio_seq[start_ent_tok_idx + 1:]:
                if next_token.split("-")[0] == "I" and next_token.split("-")[1] == entity_type:
                    end_ent_tok_idx += 1
                else:
                    break
        out_spans.append(dict(token_start=start_ent_tok_idx, token_end=end_ent_tok_idx - 1, label=entity_type))
    return out_spans


class BertNERPL(pl.LightningModule):

    def __init__(self, config: Dict, id2tag: Dict[int, str], n_training_steps: int, pretrained_bert: BertModel = None):
        super(BertNERPL, self).__init__()
        # === 1. Set main variables ==== #

        self.run_name = config['run_name']
        self.weighted_loss = assign_property(inp_config=config, parameter_name='weighted_loss', alternative=False)
        self.scaling_dict = assign_property(inp_config=config, parameter_name='scaling_dict', alternative=None)
        self.out_path = config['output_dir']
        self.id2tag = id2tag
        self.nl = len(self.id2tag)
        if "PAD" in self.id2tag.values():
            self.nl -= 1
        self.n_training_steps = n_training_steps
        # === 2. Set main hyperparameters === #
        self.seq_len = config['max_length']
        self.lr = config['learning_rate']
        self.eps = config['eps']
        #    self.weight_decay = config['weight_decay']
        self.lr_warmup = assign_property(inp_config=config, parameter_name='lr_warmup', alternative=False)
        self.weight_decay = assign_property(inp_config=config, parameter_name='weight_decay', alternative=False)
        # === 3. Load model === #
        # self.model = load_model(model_path=config['base_model'], num_labels=self.nl)
        if pretrained_bert:
            self.bert = pretrained_bert
        else:
            self.bert = AutoModel.from_pretrained(config['base_model'])
        self.dropout = torch.nn.Dropout(0.1)  # config['dropout_prob']
        self.ner_classifier = torch.nn.Linear(in_features=768,
                                              out_features=self.nl)
        self.save_hyperparameters()

    def forward(self, input_ids: torch.Tensor, attention_masks: torch.Tensor):
        """
        Adapted from https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html
        """
        outputs = self.bert(input_ids,
                            attention_mask=attention_masks)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        return self.ner_classifier(sequence_output)

    def training_step(self, inp_batch, batch_nb):

        batch_logits = self(input_ids=inp_batch['input_ids'], attention_masks=inp_batch['attention_mask'])

        loss = self.compute_ner_loss(ner_logits=batch_logits, ner_labels=inp_batch['labels'],
                                     inp_attention_masks=inp_batch['attention_mask'])

        # outputs = self.model(inp_batch['input_ids'], inp_batch['attention_mask'], labels=inp_batch['labels'],
        #                      output_hidden_states=False)  # set to true for relation classifier
        # loss = outputs[0]
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('train_loss', train_loss, prog_bar=True)

    def validation_step(self, val_batch, batch_nb):

        batch_logits = self(input_ids=val_batch['input_ids'], attention_masks=val_batch['attention_mask'])

        val_loss = self.compute_ner_loss(ner_logits=batch_logits, ner_labels=val_batch['labels'],
                                         inp_attention_masks=val_batch['attention_mask'])

        # outputs = self.model(batch['input_ids'], batch['attention_mask'], labels=batch['labels'])
        # val_loss = outputs[0]
        precision_strict, recall_strict, f1_strict, precision_partial, recall_partial, f1_partial = \
            self.compute_ner_f1s(
                predictions=batch_logits,
                labels=val_batch['labels'],
                id2tag=self.id2tag)

        return {'val_loss': val_loss,
                'val_f1_strict': f1_strict, 'val_precision_strict': precision_strict,
                'val_recall_strict': recall_strict,
                'val_f1_partial': f1_partial, 'val_precision_partial': precision_partial,
                'val_recall_partial': recall_partial}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        val_f1_strict = torch.stack([x['val_f1_strict'] for x in outputs]).mean()
        val_precision_strict = torch.stack([x['val_precision_strict'] for x in outputs]).mean()
        val_recall_strict = torch.stack([x['val_recall_strict'] for x in outputs]).mean()

        val_f1_partial = torch.stack([x['val_f1_partial'] for x in outputs]).mean()
        val_precision_partial = torch.stack([x['val_precision_partial'] for x in outputs]).mean()
        val_recall_partial = torch.stack([x['val_recall_partial'] for x in outputs]).mean()

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_f1_strict', val_f1_strict, prog_bar=True)
        self.log('val_precision_strict', val_precision_strict, prog_bar=True)
        self.log('val_recall_strict', val_recall_strict, prog_bar=True)
        self.log('val_f1_partial', val_f1_partial, prog_bar=True)
        self.log('val_precision_partial', val_precision_partial, prog_bar=True)
        self.log('val_recall_partial', val_recall_partial, prog_bar=True)

    def configure_optimizers(self):
        if self.weight_decay:
            optimizer = AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=self.eps,
                              correct_bias=False, weight_decay=self.weight_decay)
        else:
            optimizer = AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=self.eps,
                              correct_bias=False)

        if self.lr_warmup:
            scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=int(
                                                            round(self.n_training_steps * self.lr_warmup)
                                                        ),
                                                        num_training_steps=self.n_training_steps)
            return [optimizer], [scheduler]
        return optimizer

    def compute_ner_loss(self, ner_logits: torch.Tensor, ner_labels: torch.Tensor, inp_attention_masks: torch.Tensor):
        """
        Function adapted from https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#
        BertForTokenClassification
        @param ner_logits: Tensor with batch_size * seq_length * n_labels
        @param ner_labels: Tensor with batch_size * seq_length (list of integers with label id)
        @param inp_attention_masks: attention masks of the input sequence
        @return: Cross Entropy loss
        """
        if self.weighted_loss:
            weights_list = torch.tensor([self.scaling_dict[self.id2tag[i]] for i in range(0, self.nl)]).to(self.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights_list)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        active_loss = inp_attention_masks.view(-1) == 1
        active_logits = ner_logits.view(-1, self.nl)
        active_labels = torch.where(
            active_loss, ner_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(ner_labels)
        )
        return loss_fct(active_logits, active_labels)

    @staticmethod
    def compute_ner_f1s(predictions: torch.Tensor, labels: torch.Tensor, id2tag: Dict[int, str]):
        """
        @param predictions: Input tensor resulting from the softmax layer; batch_size * sequence_length * n_classes
        @param labels: Sequence length; batch_size * sequence_length * n_classes
        @param id2tag: Dictionary mapping label ids to BIO/BILOU schema
        @return: strict F1 score vs partial F1 score
        """

        # Remove labels with -100

        predictions = predictions.argmax(dim=2)
        assert predictions.shape == labels.shape

        true_predictions = [
            [id2tag[token_prediction] for (token_prediction, token_label) in zip(sentence_pred, sentence_lab) if
             token_label != -100]
            for sentence_pred, sentence_lab in zip(predictions.tolist(), labels.tolist())
        ]

        true_labels = [
            [id2tag[token_label] for (token_prediction, token_label) in zip(sentence_pred, sentence_lab) if
             token_label != -100]
            for sentence_pred, sentence_lab in zip(predictions.tolist(), labels.tolist())
        ]

        evaluator = Evaluator(true_labels, true_predictions, tags=['PK'], loader="list")
        _, results_agg = evaluator.evaluate()

        precision_strict, recall_strict, f1_strict = get_metrics(results_agg['PK']['strict'])
        precision_partial, recall_partial, f1_partial = get_metrics(results_agg['PK']['partial'])
        return precision_strict, recall_strict, f1_strict, precision_partial, recall_partial, f1_partial


def assign_property(inp_config: Dict, parameter_name: str, alternative):
    """Assigns property if exists in dictionary, otherwise returns alternative"""
    if parameter_name in inp_config.keys():
        return inp_config[parameter_name]
    return alternative


def load_pretrained_model(model_checkpoint_path, gpu):
    device = 'cpu'
    if gpu:
        device = 'cuda'

    return BertNERPL.load_from_checkpoint(
        checkpoint_path=model_checkpoint_path,
        map_location=torch.device(device),

    )


def get_metrics(inp_dict):
    p = inp_dict['precision']
    r = inp_dict['recall']
    if "f1" in inp_dict.keys():
        f1 = inp_dict['f1']
    else:
        f1 = get_f1(p=p, r=r)
    return torch.FloatTensor([p]), torch.FloatTensor([r]), torch.FloatTensor([f1])


def get_f1(p, r):
    if p + r == 0.:
        return 0.
    else:
        return (2 * p * r) / (p + r)


def clean_instance_span(instance_spans):
    return [dict(start=x['start'], end=x['end'], label=x['label']) for x in instance_spans]
