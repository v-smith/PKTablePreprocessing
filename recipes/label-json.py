import prodigy
from prodigy.components.loaders import JSONL

OPTIONS = [
    {"id": 1, "text": "Doses"},
    {"id": 2, "text": "Number of Subjects"},
    {"id": 3, "text": "Samples Timings"},
    {"id": 4, "text": "Demographics"},
    {"id": 5, "text": "Other"},
    {"id": 6, "text": "Not Relevant"},

]

@prodigy.recipe("label-json")
def label_json(dataset, html_path):
    """Stream in json tables from a directory and label them from fixed field"""

    return {
        "dataset": dataset,
        "stream": list(get_stream(html_path)),
        "view_id": "blocks",
        "config": {
            "show_flag": True,
            "choice_style": "multiple",  # or "single"
            # Automatically accept and submit the answer if an option is
            # selected (only available for single-choice tasks)
            # "choice_auto_accept": True,
            "feed_overlap": True,
            "force_stream_order": True,
            "global_css": ".prodigy-button-reject, .prodigy-button-ignore {display: none}",
            "custom_theme": {"cardMinWidth": 300, "cardMaxWidth": 1500, "smallText": 15, "show_flag": True},
            #"instructions": "./recipes/vicky/label-json-instructions.html",
            "blocks": [
                {"view_id": "choice"},
                {"view_id": "text_input", "field_rows": 3, "field_label": "Please write any comments here"},
            ]
        }
    }


def get_stream(html_path):
    # Load the directory of images and add options to each task

    stream = JSONL(html_path)

    for eg in stream:
        eg["options"] = OPTIONS
        yield eg
