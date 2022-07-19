import os
from pathlib import Path

tokens = {"tokens_all": "", "tokens_noun": "", "tokens_verb_noun": ""}


def combine_tokens(rootdir):
    for path in Path(rootdir).iterdir():  # iterate though all subdirs in rootdir
        if path.is_dir():
            combine_tokens(path)  # call this function on the subdir
        if path.is_file() and Path(path).suffix == '.txt':
            with open(path) as f:
                contents = f.read()
                # Some final cleanup
                # Some incorrect character name spellings which weren't recognized as proper nouns, removing manually
                for weird_name in ["achille", "jeeve", "holme", "achillie"]:
                    contents = contents.replace(weird_name, "")
                    contents = contents.replace("  ", " ")
                tokens[str(path).split("/")[-1].split(".")[-2]] += f" {contents}"


combine_tokens(f"{os.getcwd()}/source/input/all_data/booknlp_output")

for tokens_name, tokens_content in tokens.items():
    with open(f'./source/input/combined/{tokens_name}.txt', 'w') as f:
        f.write(tokens_content)
