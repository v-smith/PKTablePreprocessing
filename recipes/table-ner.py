from typing import Union, Optional, List
import prodigy
from prodigy.recipes.ner import manual as ner_manual
from prodigy.util import get_labels, split_string
import pickle


@prodigy.recipe("table_ner",
                dataset=("Dataset to save annotations to", "positional", None, str),
                spacy_model=(
                        "Loadable spaCy model for tokenization or blank:lang (e.g. blank:en)", "positional", None, str),
                source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
                loader=("Loader (guessed from file extension if not set)", "option", "lo", str),
                label=(
                        "Comma-separated label(s) to annotate or text file with one label per line", "option", "l",
                        get_labels),
                patterns=("Path to match patterns file", "option", "pt", str),
                exclude=(
                        "Comma-separated list of dataset IDs whose annotations to exclude", "option", "e",
                        split_string),
                highlight_chars=("Allow highlighting individual characters instead of tokens", "flag", "C", bool),
                )
def table_ner(dataset: str,
              spacy_model: str,
              source,
              loader: Optional[str] = None,
              label: Optional[List[str]] = None,
              patterns: Optional[str] = None,
              exclude: Optional[List[str]] = None,
              highlight_chars: bool = False):
    components = ner_manual(dataset=dataset, spacy_model=spacy_model, source=source,
                            loader=loader, label=label, patterns=patterns, exclude=exclude,
                            highlight_chars=highlight_chars)

    components["view_id"] = "blocks"
    components["config"]["blocks"] = [
        {"view_id": "ner_manual", "ner_manual_highlight_chars": True, "fontWeight": "bold"},
        #{"view_id": "html",
         #"html_template": """<div style="width:1450px;height:50px;border:1px solid #000;"><h4>{{text_label}}</h4></div>"""},
        {"view_id": "html", "smallText": 20}]
    components["config"]["labels"] = ["PK Param", "Units", "Mean", "Estimate", "RSE", "IIV", "Residual Err.", "Chemical/drug", "Species", "Dose",
                                      "N. Subjects", "Demographics", "Sample Timings", "Admin R."]
    components["config"]["show_flag"] = True
    components["config"]["feed_overlap"] = True
    components["config"]["force_stream_order"] = True
    components["config"]["global_css"] = ".prodigy-button-reject, .prodigy-button-ignore {display: none}"
    components["config"]["custom_theme"] = {"cardMinWidth": 300, "cardMaxWidth": 1500, "show_flag": False}
    components["global_css"]: ".prodigy-content {font-size: 30px}"

    components["config"]["instructions"] = "./recipes/vicky/table-ner-instructions.html"
    components["config"]["batch_size"] = 5
    components["config"]["history_size"] = 5

    output_stream = []
    my_pickle = pickle.load(open('./data/vicky/table_hashes_2.pkl', 'rb'))
    for eg in components["stream"]:
        hash = eg["html"]
        html = my_pickle[hash]
        eg["html"] = html
        output_stream.append(eg)

    components["stream"] = list(output_stream)

    return components


'''
Notes:
-use ner_manual for fully flexible spans
-ner manual with --patterns to help highlight some spans 
-ner.correct -- use an existing model to highlight the spans and correct it, and can add new categories on top 
'''

'''
#dataloader code 
def get_stream(html_path):
    # Load the directory of images and add options to each task
    res = JSONL(html_path)
    for eg in res:
        yield {"text": eg["text"], "text_label": eg["text_label"], "html": eg["html"]}

nlp = spacy.blank(lang)
stream = list(get_stream(html_path))
stream = add_tokens(nlp, stream)
'''
