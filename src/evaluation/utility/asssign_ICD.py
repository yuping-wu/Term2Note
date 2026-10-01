import os
import pandas as pd
import numpy as np

def get_ICD9_category(icd_code):
    """Helper function to determine the high-level category for an ICD-9 code."""
    # Remove decimals for consistent comparison
    code = icd_code.split('.')[0]
    
    # Define category ranges
    categories = [
        ((1, 139), "Infectious And Parasitic Diseases"),
        ((140, 239), "Neoplasms"),
        ((240, 279), "Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders"),
        ((280, 289), "Diseases Of The Blood And Blood-Forming Organs"),
        ((290, 319), "Mental Disorders"),
        ((320, 389), "Diseases Of The Nervous System And Sense Organs"),
        ((390, 459), "Diseases Of The Circulatory System"),
        ((460, 519), "Diseases Of The Respiratory System"),
        ((520, 579), "Diseases Of The Digestive System"),
        ((580, 629), "Diseases Of The Genitourinary System"),
        ((630, 679), "Complications Of Pregnancy, Childbirth, And The Puerperium"),
        ((680, 709), "Diseases Of The Skin And Subcutaneous Tissue"),
        ((710, 739), "Diseases Of The Musculoskeletal System And Connective Tissue"),
        ((740, 759), "Congenital Anomalies"),
        ((760, 779), "Certain Conditions Originating In The Perinatal Period"),
        ((780, 799), "Symptoms, Signs, And Ill-Defined Conditions"),
        ((800, 999), "Injury And Poisoning")
    ]
    ICD9TOCOMB = {
        'Infectious And Parasitic Diseases': 'Certain Infectious And Parasitic Diseases', 
        'Neoplasms': 'Neoplasms', 
        'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders': 'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders', 
        'Diseases Of The Blood And Blood-Forming Organs': 'Diseases Of The Blood And Blood-Forming Organs And Certain Disorders Involving The Immune Mechanism', 
        'Mental Disorders': 'Mental And Behavioural Disorders', 
        'Diseases Of The Nervous System And Sense Organs': 'Diseases Of The Nervous System And Sense Organs', 
        'Diseases Of The Circulatory System': 'Diseases Of The Circulatory System', 
        'Diseases Of The Respiratory System': 'Diseases Of The Respiratory System', 
        'Diseases Of The Digestive System': 'Diseases Of The Digestive System', 
        'Diseases Of The Genitourinary System': 'Diseases Of The Genitourinary System', 
        'Complications Of Pregnancy, Childbirth, And The Puerperium': 'Complications Of Pregnancy, Childbirth, And The Puerperium', 
        'Diseases Of The Skin And Subcutaneous Tissue': 'Diseases Of The Skin And Subcutaneous Tissue', 
        'Diseases Of The Musculoskeletal System And Connective Tissue': 'Diseases Of The Musculoskeletal System And Connective Tissue', 
        'Congenital Anomalies': 'Congenital Malformations, Deformations And Chromosomal Abnormalities', 
        'Certain Conditions Originating In The Perinatal Period': 'Certain Conditions Originating In The Perinatal Period', 
        'Symptoms, Signs, And Ill-Defined Conditions': 'Symptoms, Signs And Abnormal Clinical And Laboratory Findings, Not Elsewhere Classified', 
        'Injury And Poisoning': 'Injury, Poisoning And Certain Other Consequences Of External Causes', 
        'External Causes Of Injury And Poisoning': 'External Causes Of Morbidity And Mortality, Injusy and Poisoning', 
        'Factors Influencing Health Status And Contact With Health Services': 'Factors Influencing Health Status And Contact With Health Services'
    }
    
    # Handle special cases for E and V codes
    if code.startswith('E'):
        return ICD9TOCOMB["External Causes Of Injury And Poisoning"]
    elif code.startswith('V'):
        return ICD9TOCOMB["Factors Influencing Health Status And Contact With Health Services"]
    
    # Convert code to integer for comparison
    try:
        code_num = int(code)
        for (range_start, range_end), category in categories:
            if range_start <= code_num <= range_end:
                return ICD9TOCOMB[category]
    except ValueError:
        return None
    
    return None

def get_ICD10_category(icd_code):
    """Helper function to determine the high-level category for an ICD-10 code."""
    code = icd_code.split('.')[0]

    chapters = [
        (("A00", "B99"), "Certain Infectious And Parasitic Diseases"),
        (("C00", "D48"), "Neoplasms"),
        (("D50", "D89"), "Diseases Of The Blood And Blood-Forming Organs And Certain Disorders Involving The Immune Mechanism"),
        (("E00", "E90"), "Endocrine, Nutritional And Metabolic Diseases"),
        (("F00", "F99"), "Mental And Behavioural Disorders"),
        (("G00", "G99"), "Diseases Of The Nervous System"),
        (("H00", "H59"), "Diseases Of The Eye And Adnexa"),
        (("H60", "H95"), "Diseases Of The Ear And Mastoid Process"),
        (("I00", "I99"), "Diseases Of The Circulatory System"),
        (("J00", "J99"), "Diseases Of The Respiratory System"),
        (("K00", "K93"), "Diseases Of The Digestive System"),
        (("L00", "L99"), "Diseases Of The Skin And Subcutaneous Tissue"),
        (("M00", "M99"), "Diseases Of The Musculoskeletal System And Connective Tissue"),
        (("N00", "N99"), "Diseases Of The Genitourinary System"),
        (("O00", "O99"), "Pregnancy, Childbirth And The Puerperium"),
        (("P00", "P96"), "Certain Conditions Originating In The Perinatal Period"),
        (("Q00", "Q99"), "Congenital Malformations, Deformations And Chromosomal Abnormalities"),
        (("R00", "R99"), "Symptoms, Signs And Abnormal Clinical And Laboratory Findings, Not Elsewhere Classified"),
        (("S00", "T98"), "Injury, Poisoning And Certain Other Consequences Of External Causes"),
        (("V01", "Y98"), "External Causes Of Morbidity And Mortality"),
        (("Z00", "Z99"), "Factors Influencing Health Status And Contact With Health Services"),
        (("U00", "U99"), "Codes For Special Purposes"),
    ]
    
    ICD10TOCOMB = {
        'Certain Infectious And Parasitic Diseases': 'Certain Infectious And Parasitic Diseases', 
        'Neoplasms': 'Neoplasms', 
        'Endocrine, Nutritional And Metabolic Diseases': 'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders', 
        'Diseases Of The Blood And Blood-Forming Organs And Certain Disorders Involving The Immune Mechanism': 'Diseases Of The Blood And Blood-Forming Organs And Certain Disorders Involving The Immune Mechanism', 
        'Mental And Behavioural Disorders': 'Mental And Behavioural Disorders', 
        'Diseases Of The Nervous System': 'Diseases Of The Nervous System And Sense Organs', 
        'Diseases Of The Circulatory System': 'Diseases Of The Circulatory System', 
        'Diseases Of The Respiratory System': 'Diseases Of The Respiratory System', 
        'Diseases Of The Digestive System': 'Diseases Of The Digestive System', 
        'Diseases Of The Genitourinary System': 'Diseases Of The Genitourinary System', 
        'Pregnancy, Childbirth And The Puerperium': 'Complications Of Pregnancy, Childbirth, And The Puerperium', 
        'Diseases Of The Skin And Subcutaneous Tissue': 'Diseases Of The Skin And Subcutaneous Tissue', 
        'Diseases Of The Musculoskeletal System And Connective Tissue': 'Diseases Of The Musculoskeletal System And Connective Tissue', 
        'Congenital Malformations, Deformations And Chromosomal Abnormalities': 'Congenital Malformations, Deformations And Chromosomal Abnormalities', 
        'Certain Conditions Originating In The Perinatal Period': 'Certain Conditions Originating In The Perinatal Period', 
        'Symptoms, Signs And Abnormal Clinical And Laboratory Findings, Not Elsewhere Classified': 'Symptoms, Signs And Abnormal Clinical And Laboratory Findings, Not Elsewhere Classified', 
        'Injury, Poisoning And Certain Other Consequences Of External Causes': 'Injury, Poisoning And Certain Other Consequences Of External Causes', 
        'External Causes Of Morbidity And Mortality': 'External Causes Of Morbidity And Mortality, Injusy and Poisoning', 
        'Factors Influencing Health Status And Contact With Health Services': 'Factors Influencing Health Status And Contact With Health Services', 
        'Diseases Of The Eye And Adnexa': 'Diseases Of The Eye And Adnexa', 
        'Diseases Of The Ear And Mastoid Process': 'Diseases Of The Ear And Mastoid Process', 
        'Codes For Special Purposes': 'Codes For Special Purposes'
    }
    def is_code_in_range(code, start, end):
        """Check if the code is within the specified range."""
        code_char, code_num = code[0], int(code[1:])
        start_char, start_num = start[0], int(start[1:])
        end_char, end_num = end[0], int(end[1:])
        if code_char in [start_char, end_char]:
            return start_num <= code_num <= end_num
    
    try:
        for (start, end), category in chapters:
            if is_code_in_range(code, start, end):
                return ICD10TOCOMB[category]
    except:
        # Handle invalid codes gracefully
        return None

    return None


if __name__ == "__main__":
    icd9_file = "PATH/TO/mimiciv_icd9.feather" # output from data preprocessing step 1
    icd10_file = "PATH/TO/mimiciv_icd10.feather" # output from data preprocessing step 1

    # compare ICD-* files
    all_note_df = pd.read_feather("PATH/TO/mimic-iv-note/2.2/note/discharge.feather")
    icd9_note_df = pd.read_feather(icd9_file)
    icd10_note_df = pd.read_feather(icd10_file)

    # load SNOMED CT Entity Challenge notes
    snomed_df = pd.read_csv("PATH/TO/snomed-ct-entity-challenge/1.0.0/mimic-iv_notes_training_set.csv")
    print(f"--------#Notes in SNOMED CT Entity Challenge--------")
    print(f"{len(snomed_df)} notes \t {len(snomed_df['note_id'].unique())} unique notes")

    # check if all notes in SNOMED CT Entity Challenge are in MIMIC-IV
    snomed_note_ids = set(snomed_df['note_id'])
    all_note_ids = set(all_note_df['note_id'])
    print(f"--------#Notes in SNOMED CT Entity Challenge but not in MIMIC-IV--------")
    for note_id in snomed_note_ids:
        if note_id not in all_note_ids:
            print(note_id)
        else:
            # check if notes are the same
            snomed_note = snomed_df[snomed_df['note_id'] == note_id]
            mimic_iv_note = all_note_df[all_note_df['note_id'] == note_id]
            if snomed_note['text'].iloc[0] != mimic_iv_note['text'].iloc[0]:
                print(f"Note not equal {note_id}")

    snomed_note_ids = set(snomed_df['note_id'])
    icd9_note_ids = set(icd9_note_df['note_id'])
    icd10_note_ids = set(icd10_note_df['note_id'])
    cnt = 0
    print(f"--------#Notes in SNOMED CT Entity Challenge but not in MIMIC-IV ICD codings--------")
    for note_id in snomed_note_ids:
        if note_id not in icd9_note_ids and note_id not in icd10_note_ids:
            cnt += 1
    print(cnt)

    # combine snomed_df and icd_note_df with left join on note_id, only keep named columns from icd_note_df
    icd9_note_df.rename(columns={"target": "ICD9"}, inplace=True)
    icd10_note_df.rename(columns={"target": "ICD10"}, inplace=True)
    snomed_df_icd = pd.merge(pd.merge(snomed_df, icd9_note_df[['note_id', 'ICD9']], on='note_id', how='left'), icd10_note_df[['note_id', 'ICD10']], on='note_id', how='left')

    # check number of empty target rows 
    empty_target = snomed_df_icd[snomed_df_icd['ICD9'].isnull() & snomed_df_icd['ICD10'].isnull()]
    print(f"--------#Empty target rows--------")
    print(f"{len(empty_target)} empty target rows")
    print(snomed_df_icd)
    
    snomed_df_icd["labels"] = ""
    # for each row in snomed_df_icd, get the icd category and add it to a new column
    for i in range(len(snomed_df_icd)):
        if type(snomed_df_icd.loc[i, "ICD10"]) is np.ndarray or not pd.isnull(snomed_df_icd.loc[i, "ICD10"]):
            icd_category_func = get_ICD10_category
            codes = list(snomed_df_icd.loc[i, "ICD10"])
        elif type(snomed_df_icd.loc[i, "ICD9"]) is np.ndarray or not pd.isnull(snomed_df_icd.loc[i, "ICD9"]):
            icd_category_func = get_ICD9_category
            codes = list(snomed_df_icd.loc[i, "ICD9"])
        else:
            raise ValueError("Invalid ICD version. Choose either 'ICD9' or 'ICD10'.")
        # if not type(snomed_df_icd.loc[i, "ICD9"]) is np.ndarray and pd.isnull(snomed_df_icd.loc[i, "ICD9"]):
        #     continue
        code_categories = set()
        for code in codes:
            category = icd_category_func(code)
            if category:
                code_categories.add(category)
        snomed_df_icd.loc[i, "labels"] = "|".join(list(code_categories))
    
    # save the dataframe to a csv file
    snomed_df_icd.to_csv(f"PATH/TO/snomed-ct-entity-challenge/1.0.0/mimic-iv_notes_training_set_comb.csv", index=False)
    empty_label = snomed_df_icd[snomed_df_icd['labels'].isnull()]
    print(f"--------#Empty labels rows--------")
    print(f"{len(empty_label)} empty label rows")

    import random
    random.seed(42)
    # from sklearn.utils import shuffle
    # snomed_df_icd_shuffled = shuffle(snomed_df_icd)

    print(f"Splitting train/test dataset...")
    print(f"Full size: {len(snomed_df_icd)}")

    # split to 5 folders
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_cnt = 0
    for train_index, test_index in kf.split(snomed_df_icd):
        train, test = snomed_df_icd.iloc[train_index], snomed_df_icd.iloc[test_index]
        print(f"Train size: {len(train)}")
        print((f"Test size: {len(test)}"))
        fold_cnt += 1
        train_file = f"../../data/downstream/folds/train_comb_fold{fold_cnt}.csv"
        test_file = f"../../data/downstream/folds/test_comb_fold{fold_cnt}.csv"
        train.to_csv(train_file, index=False)
        test.to_csv(test_file, index=False)