import re
import pandas as pd
import json
from tqdm import tqdm
from multiprocessing.pool import Pool

HEADINGS = [
    "Name:",
    "Unit No:",
    "Admission Date:",
    "Discharge Date:",
    "Date of Birth:",
    "Sex:",
    "Service:",
    "Allergies:",
    "Attending:",
    "Chief Complaint:",
    "History of Present Illness:",
    "Past Medical History:",
    "Social History:",
    "Family History:",
    "Physical Exam:",
    "Pathology:",
    "Brief Hospital Course:",
    "Medications on Admission:",
    "Discharge Medications:",
    "Discharge Disposition:",
    "Discharge Diagnosis:",
    "Discharge Condition:",
    "Discharge Instructions:",
    "Followup Instructions:",
    "Discharge:",
    "Pertinent Results:",
    "Studies:",
    "Pending Results:",
    "Transitional Issues:",
    "PAST SURGICAL HISTORY:",
    "ADMISSION PHYSICAL EXAM:",
    "DISCHARGE PHYSICAL EXAM:",
    "PERTINENT LABS:",
    "DISCHARGE LABS:",
    "MICROBIOLOGY:",
    "IMAGING:",
    "ACTIVE ISSUES:",
    "CHRONIC ISSUES:",
    "Review of Systems:",
    "Major Surgical or Invasive Procedure:",
    "ADMISSION CXR:",
    "FOLLOW UP CXR:",
    "VASCULAR SURGERY ADMISSION EXAM:",
    "ADMISSION LABS:",
    "DEATH EXAM:",
    "CXR:",
    "CXR ___:",
    "SECONDARY:",
    "LABS:"
]

def check_continuity(d, node_id, total):
    missing_lens = 0
    last_key = None
    for k, v in d.items():
        if last_key is None:
            last_key = k
        else:
            try:
                assert v[0] >= d[last_key][1]
            except AssertionError:
                print(f"Potential wrong segmentation between {last_key} and {k} in note {node_id}")
            missing_lens += max(0, v[0] - d[last_key][1])
            last_key = k
    missing_lens += max(0, total - d[last_key][1])
    return missing_lens


'''
Parameters and Functions to extract HEADING content and group to sections
'''
HEADINGS_GROUP = {
    "Patient Information": [
        "Name", "Unit No", "Admission Date", "Discharge Date", "Date of Birth", "Sex", 
        "Service", "Allergies", "Attending"
    ],
    "Clinical Course & History":[
        "Chief Complaint", "Major Surgical or Invasive Procedure", "History of Present Illness", 
        "Review of Systems", "Past Medical History", "Social History", "Family History"
    ],
    "Examinations & Findings": [
        "Physical Exam"
    ],
    "Laboratory & Imaging Results": [
        "Pertinent Results"
    ],
    "Hospital Stay & Treatment": [
        "Brief Hospital Course"
    ],
    "Medications & Discharge Plan": {
        "Medications on Admission", "Discharge Medications", 
        "Discharge Disposition", "Discharge Diagnosis", "Discharge Condition", 
        "Discharge Instructions", "Followup Instructions"
    }
}

def head2sec():
    # convert hierarchy to heading -> sections
    heading2section = {}
    for section in HEADINGS_GROUP.keys():
        for heading in HEADINGS_GROUP[section]:
            heading2section[heading] = section
    HEADINGS = list(heading2section.keys())
    return heading2section, HEADINGS

def extract_group_sections(params):
    note, note_id, heading2section, HEADINGS = params  
    section_dict = {}
    # for heading in known_headings:
    for heading in HEADINGS:
        # print(f"Extracting section for heading: {heading}")
        pattern = r"(^|\s\s+)" + re.escape(heading+":")

        if not re.search(pattern, note):
            continue

        match = re.search(pattern, note)

        start_index_extract = match.start()

        # find closest next section, starting from end of note
        next_section_index = len(note) - 1
        for next_heading in HEADINGS:
            if next_heading.__eq__(heading):
                continue

            pattern_next = r"(^|\s\s+)" + re.escape(next_heading)
            match_next = re.search(pattern_next, note)

            if not re.search(pattern_next, note):
                continue

            if next_section_index > match_next.start() > start_index_extract:
                next_section_index = match_next.start()

        # extract section between start and next section, store
        section_dict[heading] = [start_index_extract, next_section_index, note[start_index_extract:next_section_index]]
    
    sorted_section_dict = {k: v for k, v in sorted(section_dict.items(), key=lambda item: item[1][0])}

    # group headings to sections
    grouped_section_dict = {}
    last_section = ""
    for heading, value in sorted_section_dict.items():
        section = heading2section[heading]
        if section != last_section and section not in grouped_section_dict:
            grouped_section_dict[section] = value
            last_section = section
        else:
            grouped_section_dict[section][1] = value[1]
            grouped_section_dict[section][2] += value[2]
    
    # check content consistency and continuity
    for sec, value in grouped_section_dict.items():
        if value[2] != note[value[0]:value[1]]:
            print(f"Content inconsistency in note {note_id} section {sec}")
    missing_length = check_continuity(grouped_section_dict, note_id, len(note)-1)
    return note_id, grouped_section_dict, missing_length

if __name__ == '__main__':
    input_file = "../../data/path-to-mimiciv_icd10.csv"
    output_file = "../../data/path-to-mimiciv_icd10_sections.json"
    NUM_THREADS = 16

    # load notes
    df = pd.read_csv(input_file)

    heading2section, headings = head2sec() # for group task

    # iterate over notes
    todo = []
    processed_sections = {}
    for i in range(len(df)):
        nid = df.loc[i, 'note_id']
        note = df.loc[i, 'text']
        # todo.append([note, nid])
        todo.append([note, nid, heading2section, headings]) # for group task
        processed_sections[nid] = {}
        processed_sections[nid]["full_note"] = note

    # extract sections
    num_valid_headings, missing_lens = [], []
    _p = Pool(NUM_THREADS)
    for result in tqdm(_p.imap_unordered(extract_group_sections, todo), total=len(todo)):
        nid, sections, missing_len = result
        processed_sections[nid]["missing_length"] = missing_len
        processed_sections[nid]["sections"] = sections
        num_valid_headings.append(len(sections))
        missing_lens.append(missing_len)
    _p.close()
    _p.join()

    print(f"Number of notes: {len(df)}")
    print(f"Average number of valid headings: {sum(num_valid_headings)/len(num_valid_headings)}")
    print(f"Average number of missing lengths: {sum(missing_lens)/len(missing_lens)}")

    # save
    with open(output_file, 'w') as f:
        json.dump(processed_sections, f, indent=4)