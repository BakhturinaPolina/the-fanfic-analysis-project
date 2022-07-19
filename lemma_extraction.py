import csv
import os
import re
from pathlib import Path

import pandas as pd
import spacy

# Load the English Language Model of Spacy.
spacy_client = spacy.load('en_core_web_sm')

def lemmatize_and_remove_names(path):
    df_tokens = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE)
    tokens_all = df_tokens.loc[(df_tokens['POS_tag'] != "PROPN") & (df_tokens['POS_tag'] != "PUNCT")]
    tokens_verb_noun = df_tokens.loc[(df_tokens['POS_tag'] == "VERB") | (df_tokens['POS_tag'] == "NOUN")]
    tokens_noun = df_tokens.loc[(df_tokens['POS_tag'] == "NOUN")]

    data_variants = {"tokens_all": tokens_all, "tokens_verb_noun": tokens_verb_noun, "tokens_noun": tokens_noun}

    lemmas = {}
    book_path = "/".join(str(path).split("/")[:-1])

    # book_id = "".join(str(path).split("/")[-1:]).split('.')[0]

    for data_variant_name, data_variant in data_variants.items():
        lemmas[data_variant_name] = data_variant["lemma"].tolist()
        lemmas[data_variant_name] = [str(value).lower() for value in lemmas[data_variant_name]]
        lemma_string = " ".join(lemmas[data_variant_name])
        lemmas[data_variant_name] = lemma_string.replace("--", " ")
        lemmas[data_variant_name] = re.sub(r'[^A-Za-z ]+', '', lemma_string)
        lemmas[data_variant_name] = re.sub('\s+', ' ', lemmas[data_variant_name])
        lemmas[data_variant_name] = lemmas[data_variant_name].split()
        lemmas[data_variant_name] = [word for word in lemmas[data_variant_name] if
                                     word not in spacy_client.Defaults.stop_words and "www" not in word and len(
                                         word) > 2]
        with open(f"{book_path}/{data_variant_name}.txt", "w") as text_file:
            text_file.write(" ".join(lemmas[data_variant_name]))


def process_all_texts(rootdir):
    for path in Path(rootdir).iterdir():  # iterate though all subdirs in rootdir
        if path.is_dir():
            process_all_texts(path)  # call this function on the subdir
        if path.is_file() and Path(path).suffix == '.tokens':
            lemmatize_and_remove_names(path)


process_all_texts(f"{os.getcwd()}/source/input/all_data/booknlp_output")
