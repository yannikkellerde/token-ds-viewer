from xml.etree import ElementTree as ET
from xml.etree.ElementTree import ParseError

from datasets import Dataset
from transformers import PreTrainedTokenizer

replacements = {"▁": " ", "<0x0A>": "<br/>", "Ġ": " ", "\n": "<br>"}
IGNORE_INDEX = -100


def load_ds(path, tk: PreTrainedTokenizer, row_start=0, row_end=100):
    ds = Dataset.load_from_disk(path)
    ds = ds.select(range(row_start, row_end))

    texts = []
    for row in ds:
        texts.append(tk.convert_ids_to_tokens(row["input_ids"]))

    return ds.add_column("text", texts)


def many_entries(entries, color_col=None):
    html = ET.Element("html")
    body = ET.Element(
        "body", attrib={"style": "display:flex;flex-direction:row;flex-wrap:wrap;"}
    )

    html.append(body)
    for entry in entries:
        div = html_for_entry(entry, color_col)
        body.append(div)

    return html


def html_for_entry(entry: dict, color_col=None):
    div = ET.Element(
        "div",
        attrib={
            "style": "display: inline-block; border-right: 1px solid black; padding: 10px;border-bottom: 1px solid black;width:20%;text-align:left;"
        },
    )
    for i in range(len(entry["text"])):
        tex: str = entry["text"][i]
        for old, new in replacements.items():
            tex = tex.replace(old, new)

        style = ""

        title_parts = []

        for key, value in entry.items():
            if key != "text":
                if isinstance(value, list):
                    value_elem = value[i]
                    display = True
                    if isinstance(value_elem, list):
                        display = any(x != IGNORE_INDEX for x in value)
                    elif isinstance(value_elem, int) or isinstance(value, float):
                        display = value_elem != IGNORE_INDEX
                    else:
                        assert isinstance(
                            value_elem, str
                        ), f"Unexpected type: {type(value_elem)}"
                    if display:
                        title_parts.append(f"{key}: {value_elem}")
                        if key == color_col:
                            style = "color:blue;"

        title = "&#10;".join(title_parts)
        try:
            span = ET.fromstring(f"<span title='{title}'>{tex}</span>")
        except ParseError:
            span = ET.Element("span")
            span.text = tex
            span.attrib["title"] = title
        span.attrib["style"] = style
        div.append(span)

    return div
