from xml.etree import ElementTree as ET

import fire
from transformers import AutoTokenizer, PreTrainedTokenizer

from ds_viewer.generate_html import html_for_entry, load_ds, many_entries


def show_entries(ds_path, tk_path, entry_start, entry_stop, color_col, output_path):
    tk = AutoTokenizer.from_pretrained(tk_path)
    ds = load_ds(ds_path, tk, row_start=entry_start, row_end=entry_stop)
    html = many_entries(ds, color_col)

    with open(output_path, "w") as f:
        f.write(ET.tostring(html).decode("utf-8"))


def do_show_entries():
    fire.Fire(show_entries)


if __name__ == "__main__":
    do_show_entries()
