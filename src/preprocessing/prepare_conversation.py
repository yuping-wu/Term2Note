from fire import Fire
import json
from joblib import Parallel, delayed
from tqdm import tqdm
from copy import deepcopy


FINETUNE_PROMPT_TEMPLATE = {
    "conversation_starter": [
        {
            "role": "system",
            "content": "**You are an AI assistant trained to generate structured clinical discharge notes.**",
        },
        {
            "role": "user",
            "content": """**Task:** Given a list of clinical terms, generate a "{section_name}" section in a discharge note with all the provided clinical terms incorporated. 
**Clinical Terms:** 
[{section_terms}]

**Now generate the section content for section {section_name}:**""",
        },
        {"role": "assistant", "content": "{section_text}"},
    ],
    "conversation_continue": [
        {
            "role": "user",
            "content": """**Task:** Given a list of clinical terms, generate a "{section_name}" section that builds on the previous section in a discharge note with all the provided clinical terms incorporated. 
**Clinical Terms:** 
[{section_terms}]

**Now generate the section content for section {section_name}:**""",
        },
        {"role": "assistant", "content": "{section_text}"},
    ],
}

def process_row_conversation(row, unique=True):
    # result = []
    row.pop("full_note")
    sorted_section_row = sorted([(s, r) for s, r in row.items()], key=lambda r: r[1]["indices"][0])
    (starter_name, starter_values), *rest = sorted_section_row
    starter_text = starter_values["text"]
    section_terms = ", ".join(set(t.lower() for t in starter_values["terms"]) if unique else starter_values["terms"])
    conversation_starter = deepcopy(FINETUNE_PROMPT_TEMPLATE["conversation_starter"])
    conversation_starter[1]["content"] = conversation_starter[1]["content"].format(
        section_name=starter_name,
        section_terms=section_terms,
    )
    conversation_starter[2]["content"] = conversation_starter[2]["content"].format(section_text=starter_text)
    msg_history = conversation_starter

    for continuation_name, continuation_value in rest:
        continuation_text = continuation_value["text"]
        continuation_terms = ", ".join(
            set(t.lower() for t in continuation_value["terms"]) if unique else continuation_value["terms"]
        )
        conversation_cont = deepcopy(FINETUNE_PROMPT_TEMPLATE["conversation_continue"])
        conversation_cont[0]["content"] = conversation_cont[0]["content"].format(
            section_name=continuation_name,
            section_terms=continuation_terms,
        )
        conversation_cont[1]["content"] = conversation_cont[1]["content"].format(
            section_text=continuation_text,
        )
        msg_history.extend(conversation_cont)
    return [msg_history]

def load_json(file):
    return json.load(open(file, "r"))

def write_jsonl(content, outfile):
    with open(outfile, 'w') as outf:
        for c in content:
            outf.write(json.dumps(c) + "\n")

def main(dataset="../../data/path-to-mimiciv_icd10_sections_umls_filtered.json", n_jobs=24, output_file=None, unique=True):
    process_row = process_row_conversation

    output_file = output_file or f"../../data/preprocessed.jsonl"
    print("Loading data")
    data = [v for _, v in load_json(dataset).items()]
    # process_row(data[-1])
    # print(json.dumps(process_row(data[0])))

    # print("Data loaded")
    result = Parallel(n_jobs=n_jobs)(delayed(process_row)(row, unique=unique) for row in tqdm(data))
    result = [{"messages": k} for l in result for k in l]
    write_jsonl(result, output_file)


if __name__ == "__main__":
    Fire(main)