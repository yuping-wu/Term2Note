import json
from tqdm import tqdm

umls_result_file = "../../data/path-to-mimiciv_icd10_umls.jsonl"
section_file = "../../data/path-to-mimiciv_icd10_sections.json"
out_file = "../../data/path-to-mimiciv_icd10_sections_umls_filtered.json"

# path to the filtered UMLS to SNOMED CT mapping file
cui2scui = json.load(open("PATH/TO/cui2scui.json", "r"))

sections = json.load(open(section_file, "r"))

dataset = {}
original_term_cnt, filtered_term_cnt = [], []
with open(umls_result_file, "r") as f:
    for line in tqdm(f):
        # dataset.append(json.loads(line))
        data = json.loads(line)

        note_id = data["note_id"]
        note = data["text"]

        note_sections = sections[note_id]["sections"]
        section_range = {}
        new_sections = {}
        for k, v in note_sections.items():
            sidx, eidx, text = v
            section_range[(sidx, eidx)] = k
            new_sections[k] = {
                "text": text,
                "indices": [sidx, eidx],
                "terms": []
            }
        

        umls_result = data["UMLS"]
        filtered_umls_result = []
        # filter out umls not in SNOMED CT
        for l1, l2 in umls_result:
            ngram, ngram_sidx, ngram_eidx = l1
            for term, cui, sim in l2:
                if cui in cui2scui:
                    filtered_umls_result.append(ngram)
                    for range_k, range_v in section_range.items():
                        sec_sidx, sec_eidx = range_k
                        if ngram_sidx >= sec_sidx and ngram_sidx <= sec_eidx:
                            new_sections[range_v]["terms"].append(ngram)
                            break
                    break
        
        original_term_cnt.append(len(umls_result))
        filtered_term_cnt.append(len(filtered_umls_result))
        new_data = {
            "full_note": {
                "text": note,
                "indices": [0, len(note)],
                "terms": filtered_umls_result
            },
            **new_sections
        }
        dataset[note_id] = new_data

with open(out_file, "a") as outf:
    json.dump(dataset, outf, indent=4)

print(f"Average number of terms in notes before filerting: {sum(original_term_cnt) / len(original_term_cnt)}")
print(f"Average number of terms in notes after filerting: {sum(filtered_term_cnt) / len(filtered_term_cnt)}")