from fire import Fire
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import random

def process_row(row, unique=True):
    row.pop("full_note")
    sorted_section_row = sorted([(s, r) for s, r in row.items()], key=lambda r: r[1]["indices"][0])
    section_terms = []
    for section_name, section_value in sorted_section_row:
        terms = list(set(t.lower() for t in section_value["terms"])) if unique else section_value["terms"]
        if terms:
            section_terms.append("\n".join(terms))
    return section_terms


def load_json(file):
    return json.load(open(file, "r"))


def main(dataset="../../data/path-to-mimiciii_sections_umls_filtered.json", n_jobs=12, unique=True):
    output_tranin_file = "../../data/mimiciii_terms_train.json"
    output_val_file = "../../data/mimiciii_terms_val.json"
    print("Loading data")
    data = [v for _, v in load_json(dataset).items()]
    # split the data into train and validation sets
    random.seed(42)  # For reproducibility
    random.shuffle(data)
    # split_index = int(len(data) * 0.9)
    split_index = len(data) - 500
    train_data = data[:split_index]
    val_data = data[split_index:]

    # print("Data loaded")
    train_result = Parallel(n_jobs=n_jobs)(delayed(process_row)(row, unique=unique) for row in tqdm(train_data))
    train_result = sum(train_result, [])  # Flatten the list of lists
    print(f"Processed {len(train_result)} training rows")
    train_result = set(train_result)  # Flatten and remove duplicates
    train_result = [term.split("\n") for term in train_result]  # Split terms by newline
    print(f"Unique training rows: {len(train_result)}")
    val_result = Parallel(n_jobs=n_jobs)(delayed(process_row)(row, unique=unique) for row in tqdm(val_data))
    val_result = sum(val_result, [])  # Flatten the list of lists
    print(f"Processed {len(val_result)} validation rows")
    val_result = set(val_result)  # Flatten and remove duplicates
    val_result = [term.split("\n") for term in val_result]  # Split terms by newline
    print(f"Unique validation rows: {len(val_result)}")

    # write the results to json files
    with open(output_tranin_file, 'w') as outf:
        json.dump(train_result, outf, indent=2)
    with open(output_val_file, 'w') as outf:
        json.dump(val_result, outf, indent=2)
    


if __name__ == "__main__":
    Fire(main)
    