from pathlib import Path
from booknlp.booknlp import BookNLP

model_params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "small"
}

booknlp = BookNLP("en", model_params)

output_directory = "./source/input/all_data/booknlp_output"


def process_all_texts(rootdir):
    for path in Path(rootdir).iterdir():  # iterate though all subdirs in rootdir
        if path.is_dir():
            process_all_texts(path)  # call this function on the subdir
        if path.is_file():
            book_id = Path(path).stem  # .stem returns the last element of the path, which in our case is author name
            booknlp.process(path, output_directory, book_id)
            print(f"Preprocessing book {Path(path).stem}")


process_all_texts(f"./source/input/")
