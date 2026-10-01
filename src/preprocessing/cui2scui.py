import os
import pandas as pd
import json, re

# replace with your own path to the UMLS MRCONSO.RRF file and the SNOMED-CT directory
umls_file = "/PATH/TO/2023AA-full/2023AA/META/MRCONSO.RRF"
snomed_ct_directory = "/PATH/TO/SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z/Snapshot"
out_mapping_file = "PATH/TO/cui2scui.json"


"""
Step 1: Load UMLS codes.
"""
line_cnt = 0
snomed_lines = []
with open(umls_file, "r") as f:
    for l in f:
        line_cnt += 1
        if "SNOMEDCT" in l:
            snomed_lines.append(l)
print(f"# rows in total: {line_cnt} \t # SNOMED-CT rows: {len(snomed_lines)}")

columns = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI",
    "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"
]
column_values = {col: [] for col in columns}
for line in snomed_lines:
    row = line.rstrip("\n").split("|")[:18]
    try:
        assert len(row) == 18
    except:
        print("Abnormal line:\t", line)
    for column, value in zip(columns, row):
        column_values[column].append(value)

df = pd.DataFrame.from_dict(column_values)
# df.to_pickle("../../data/UMLS/MRCONSO_SNOMEDCT.pkl")
# df = pd.read_pickle("../../data/UMLS/MRCONSO_SNOMEDCT.pkl")
# print(f"Loaded {len(df)} rows from the raw file.")

# drop rows whose SAB is not SNOMEDCT_US
df = df[df['SAB'] == "SNOMEDCT_US"].reset_index()
print(f"Remaining {len(df)} rows after filtering with source name as SNOMED_US.")
print(f"Number of empty CUI: {df['CUI'].isnull().sum()}")
print(f"Number of unique CUI: {df['CUI'].nunique()}")
print(f"Number of empty SCUI: {df['SCUI'].isnull().sum()}")
print(f"Number of unique SCUI: {df['SCUI'].nunique()}")
print(f"Number of empty CODEs: {df['CODE'].isnull().sum()}")
print(f"Number of unique CODEs: {df['CODE'].nunique()}")


"""
Step 2: Load SNOMED-CT vocabulary.
"""
# load full SNOMED-CT vocabulary
def read_file_and_subset_to_active(filename):
    with open(filename, encoding="utf-8") as f:
        # replacement of semi-colon necessary - otherwise some synonyms extracted incorrectly e.g. "Mixed
        # phenotype acute leukemia with t("
        entities = [[n.replace(";", ":").strip() for n in line.split("\t")] for line in f]
        input_df = pd.DataFrame(entities[1:], columns=entities[0])

        return input_df[input_df.active == "1"]

active_terms = read_file_and_subset_to_active(
    os.path.join(snomed_ct_directory, "Terminology", "sct2_Concept_Snapshot_INT_20230531.txt")
)
active_descs = read_file_and_subset_to_active(
    os.path.join(snomed_ct_directory, "Terminology", "sct2_Description_Snapshot-en_INT_20230531.txt")
)

snomed_df = pd.merge(active_terms, active_descs, left_on=["id"], right_on=["conceptId"], how="inner")[
    ["id_x", "term", "typeId"]
].rename(columns={"id_x": "concept_id", "term": "concept_name", "typeId": "name_type"})

# active description or active synonym
snomed_df["name_type"] = snomed_df["name_type"].replace(
    ["900000000000003001", "900000000000013009"], ["P", "A"]
)
print(f"Number of unique concepts in SNOMED CT: {snomed_df['concept_id'].nunique()}")

# map between df['CODE'] and snomed_df['concept_id']
df_codes = set(df["CODE"].unique())
df_scuis = set(df["SCUI"].unique())
snomed_codes = set(snomed_df["concept_id"].unique())
print(f"Number of unique codes in UMLS: {len(df_codes)}")
print(f"Number of unique SCUI in UMLS: {len(df_scuis)}")
print(f"Number of unique codes in SNOMED-CT: {len(snomed_codes)}")
print(f"Number of common codes: {len(df_codes.intersection(snomed_codes))}")
print(f"Number of common SCUI: {len(df_scuis.intersection(snomed_codes))}")
# 10535 in df_SCUIs but not in snomed_codes: in US edition but not in international edition OR being inactive
# 4014 in snomed_codes but not in df_SCUIs: in international edition but not in US edition


"""
Step 3: Filter SNOMED-CT concepts.
"""
# filter those SNOMED concepts belonging to "body structure", "procedure" and "finding"
# concept types to filter to
concept_types = [
    "body structure", "procedure",
    "finding",  # EXCLUDES disorders
    # "disorder",
    # "morphologic abnormality"
]
def get_hierarchy_tag(desc):
    desc = str(desc)
    hierarchy_tag = re.sub('^.*\((.*?)\)[^\(]*$', '\g<1>', desc)
    return hierarchy_tag

# load synonyms
fsns_df = snomed_df.loc[snomed_df['name_type'] == "P"]
fsns_df['hierarchy'] = fsns_df['concept_name'].apply(lambda x: re.sub('^.*\((.*?)\)[^\(]*$', '\g<1>', str(x)))
fsns_df = fsns_df.loc[fsns_df['hierarchy'].isin(concept_types)]
valid_concept_ids = list(fsns_df['concept_id'])
filtered_snomed_df = snomed_df.loc[snomed_df['concept_id'].isin(valid_concept_ids)].reset_index()


"""
Step 4: Map UMLS codes to SNOMED-CT concepts.
"""
# map between df['CODE'] and filtered_snomed_df['concept_id']
df_codes = set(df["CODE"].unique())
df_scuis = set(df["SCUI"].unique())
filtered_snomed_codes = set(filtered_snomed_df["concept_id"].unique())
print(f"Number of unique codes in UMLS: {len(df_codes)}")
print(f"Number of unique SCUI in UMLS: {len(df_scuis)}")
print(f"Number of unique codes in SNOMED-CT: {len(filtered_snomed_codes)}")
print(f"Number of common codes: {len(df_codes.intersection(filtered_snomed_codes))}")
print(f"Number of common SCUI: {len(df_scuis.intersection(filtered_snomed_codes))}")


# for those occuring in both UMLS and SNOMED-CT, map them
snomed_id2name = {}
for i in range(len(filtered_snomed_df)):
    cid = filtered_snomed_df.loc[i, "concept_id"]
    snomed_id2name[cid] = snomed_id2name.get(cid, [])
    snomed_id2name[cid].append(filtered_snomed_df.loc[i, "concept_name"])

common_codes = df_scuis.intersection(filtered_snomed_codes)
print(f"Number of rows whose SCUI is in SNOMED-CT: {len(df[df['SCUI'].isin(common_codes)])}") # 335392
cui2scui = {}
for i in range(len(df)):
    cui = df.loc[i, "SCUI"]
    if cui in common_codes:
        cui2scui[df.loc[i, "CUI"]] = [cui, snomed_id2name[cui]]
print(f"Number of CUIs whose SCUI is in SNOMED-CT: {len(cui2scui)}") # 

with open(out_mapping_file, 'w') as f:
    json.dump(cui2scui, f, indent=4)