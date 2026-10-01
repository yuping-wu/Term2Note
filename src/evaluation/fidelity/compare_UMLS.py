import json, os
from tqdm import tqdm
import numpy as np
import multiprocessing
from multiprocessing.pool import Pool
from quickumls import QuickUMLS
from collections import Counter
from scipy.stats import entropy, cumfreq
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-deep')

import pdb

NUM_BINS = 50

SECTIONS = [
    "Patient Information",
    "Clinical Course & History",
    "Examinations & Findings",
    "Laboratory & Imaging Results",
    "Hospital Stay & Treatment",
    "Medications & Discharge Plan"
]

CUI2SCUI = json.load(open("PATH/TO/cui2scui.json", "r"))
QUICKUMLS_PATH = "/PATH/TO/QUICKUMLS/DATA" # Update this path to your QuickUMLS data directory

def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

def format_umls_result(umls_result):
    new_umls_result = {}
    for result in umls_result:
        ngram = result[0]["ngram"]
        sidx, eidx = result[0]["start"], result[0]["end"]
        term_cui_sim_list = [[result[0]["term"], result[0]["cui"], result[0]["similarity"]]]

        for r in result[1:]:
            try:
                assert ngram == r["ngram"] and sidx == r["start"] and eidx == r["end"]
            except:
                print(f"Unmatched ngram/indices error for UMLS result:\n{umls_result}")
            term_cui_sim_list.append([r["term"], r["cui"], r["similarity"]])
        
        # sort by similarity
        term_cui_sim_list.sort(key=lambda x: x[2], reverse=True)

        new_umls_result[(ngram, sidx, eidx)] = term_cui_sim_list
    
    # sort by indices
    sorted_umls_result = dict(sorted(new_umls_result.items(), key=lambda x: x[0][1]))

    # check consistency
    try:
        assert len(sorted_umls_result) == len(umls_result)
    except:
        print(f"Unmatched number of ngrams error for UMLS result:\n {umls_result}")
    
    return [[k, v] for k, v in sorted_umls_result.items()]
            
def filter_snomed_terms(formated_result):
    filtered_umls_result = []
    for l1, l2 in formated_result:
        ngram, ngram_sidx, ngram_eidx = l1
        for term, cui, sim in l2:
            if cui in CUI2SCUI:
                filtered_umls_result.append([ngram, cui])
                break
    return filtered_umls_result


def extract_UMLS(params):
    data = params

    matcher = QuickUMLS(QUICKUMLS_PATH, threshold=0.5)
    all_source_terms, all_generated_terms = {}, {}
    
    # format UMLS result -- conversation format
    source_messages = data["original_messages"]
    generated_messages = data["messages"]
    for section in SECTIONS:
        source_text, generated_text = None, None

        for idx, mes in enumerate(source_messages):
            if mes["role"] == "user" and section in mes["content"]:
                assert source_messages[idx+1]["role"] == "assistant"
                source_text = source_messages[idx+1]["content"]
                break
        
        for idx, mes in enumerate(generated_messages):
            if mes["role"] == "user" and section in mes["content"]:
                assert generated_messages[idx+1]["role"] == "assistant"
                generated_text = generated_messages[idx+1]["content"]
                break
        
        if source_text and generated_text:
            source_result = matcher.match(source_text, best_match=True, ignore_syntax=False)
            source_terms = format_umls_result(source_result)
            source_terms_filtered = filter_snomed_terms(source_terms)
            generated_result = matcher.match(generated_text, best_match=True, ignore_syntax=False)
            generated_terms = format_umls_result(generated_result)
            generated_terms_filtered = filter_snomed_terms(generated_terms)

            all_source_terms[section] = source_terms_filtered # [[term, CUI]]
            all_generated_terms[section] = generated_terms_filtered

    return [all_source_terms, all_generated_terms]


def compare_cumFreq(freqT1, freqT2, T1label="original", T2label="synthetic", output_file=None):
    # plot cumulative frequency
    array1 = np.array(list(freqT1.values()))
    array2 = np.array(list(freqT2.values()))

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    
    # plot histogram
    ax1.hist([array1, array2], bins=NUM_BINS, label=[T1label, T2label], alpha=0.5)
    ax1.set_title('Frequency Histogram')
    ax1.legend()

    # plot cumulative histogram
    res1 = cumfreq(array1, numbins=NUM_BINS)
    res2 = cumfreq(array2, numbins=NUM_BINS)
    x1 = res1.lowerlimit + np.linspace(0, res1.binsize*res1.cumcount.size, res1.cumcount.size)
    x2 = res2.lowerlimit + np.linspace(0, res2.binsize*res2.cumcount.size, res2.cumcount.size)
    ax2.bar(x1, res1.cumcount, width=res1.binsize, alpha=0.5, label=T1label)
    ax2.bar(x2, res2.cumcount, width=res2.binsize, alpha=0.5, label=T2label)
    ax2.set_title('Cumulative histogram')
    ax2.set_xlim([min(x1.min(), x2.min()), max(x1.max(), x2.max())])
    ax2.legend()

    # SAVE the figure before showing it
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show() 


def aggregate_all_cuis(all_note_terms):
    all_cuis_agg = []
    for note in all_note_terms:
        for section in SECTIONS:
            if section not in note:
                continue
            all_cuis_agg += [v[1] for v in note[section]]
    return all_cuis_agg

def compare_uni_UMLS(sources, augmentations, topk=None, output_folder=None):
    print("==================================Results for unigram==================================")
    uni_result = {
        "statistics": {},
        "frequency difference": {},
        "metrics": {}
    } 

    # compare term frequency
    all_source_cuis = aggregate_all_cuis(sources)
    all_augmentation_cuis = aggregate_all_cuis(augmentations)

    source_freq_table = Counter(all_source_cuis)
    augmentation_freq_table = Counter(all_augmentation_cuis)
    
    source_cuis_set = set(source_freq_table.keys())
    augmentation_cuis_set = set(augmentation_freq_table.keys())
    jac = jaccard_similarity(source_cuis_set, augmentation_cuis_set)
    uni_result["metrics"]["jaccard similarity"] = jac
    print(f">>>>>>>>>Jaccard Similarity of terms: {float(jac)}<<<<<<<<<<<<<<<")
    # pdb.set_trace()
    diff1 = source_cuis_set - augmentation_cuis_set
    diff2 = augmentation_cuis_set - source_cuis_set
    diff1_freq = {k: source_freq_table[k] for k in diff1}
    diff1_freq_sorted = dict(sorted(diff1_freq.items(), key=lambda item: item[1], reverse=True))
    diff2_freq = {k: augmentation_freq_table[k] for k in diff2}
    diff2_freq_sorted = dict(sorted(diff2_freq.items(), key=lambda item: item[1], reverse=True))
    print(f"Number of unique terms/CUIs in original corpus: {len(source_cuis_set)}")
    uni_result["statistics"]["#source terms"] = len(source_cuis_set)
    print(f"Number of unique terms/CUIs in synthetic corpus: {len(augmentation_cuis_set)}")
    uni_result["statistics"]["#synthetic terms"] = len(augmentation_cuis_set)
    print(f"Number of unique terms/CUIs in original corpus but not in synthetic corpus: {len(diff1)} \t Max-Min occurrence {max(diff1_freq.values())}-{min(diff1_freq.values())}")
    uni_result["statistics"]["#in-source-not-in-synthetic terms"] = len(diff1)
    uni_result["statistics"]["maximum frquency of in-source-not-in-synthetic terms"] = max(diff1_freq.values())
    uni_result["statistics"]["minimum frquency of in-source-not-in-synthetic terms"] = min(diff1_freq.values())
    # print(f"Examples:\n {[[CUI2SCUI[t[0]][1][0], t[1]] for t in list(diff1_freq_sorted.items())[:5]]}")
    print(f"Number of unique terms/CUIs in synthetic corpus but not in original corpus: {len(diff2)} \t Max-Min occurrence {max(diff2_freq.values())}-{min(diff2_freq.values())}")
    uni_result["statistics"]["#in-synthetic-not-in-source terms"] = len(diff2)
    uni_result["statistics"]["maximum frquency of in-synthetic-not-in-source terms"] = max(diff2_freq.values())
    uni_result["statistics"]["minimum frquency of in-synthetic-not-in-source terms"] = min(diff2_freq.values())
    # print(f"Examples:\n {[[CUI2SCUI[t[0]][1][0], t[1]] for t in list(diff2_freq_sorted.items())[:5]]}")
    
    source_freq, augmentation_freq = [], []
    freq_diff = []
    for cui in set(source_freq_table.keys()) | set(augmentation_freq_table.keys()):
        source_freq.append(source_freq_table.get(cui, 1e-6) / len(sources))
        augmentation_freq.append(augmentation_freq_table.get(cui, 1e-6) / len(augmentations))
        freq_diff.append(abs(source_freq_table.get(cui, 0) - augmentation_freq_table.get(cui, 0)))
    
    # print statistics for freq_diff
    print(f"Min-Max occurrence difference: {min(freq_diff)}-{max(freq_diff)}")
    print(f"Mean occurrence difference: {np.mean(freq_diff)}")
    print(f"Median occurrence difference: {np.median(freq_diff)}")
    print(f"p90 occurrence difference: {np.percentile(freq_diff, 90)}")
    print(f"p95 occurrence difference: {np.percentile(freq_diff, 95)}")
    print(f"p99 occurrence difference: {np.percentile(freq_diff, 99)}")
    uni_result["frequency difference"]["maximum"] = max(freq_diff)
    uni_result["frequency difference"]["minimum"] = min(freq_diff)
    uni_result["frequency difference"]["mean"] = np.mean(freq_diff)
    uni_result["frequency difference"]["median"] = np.median(freq_diff)
    uni_result["frequency difference"]["p90"] = np.percentile(freq_diff, 90)
    uni_result["frequency difference"]["p95"] = np.percentile(freq_diff, 95)
    uni_result["frequency difference"]["p99"] = np.percentile(freq_diff, 99)
    
    kl_div = entropy(pk=source_freq, qk=augmentation_freq)
    print(f">>>>>>>>>KL divergence of term frequency: {float(kl_div)}<<<<<<<<<<<<<<<")
    uni_result["metrics"]["KL divergence"] = kl_div
    
    if topk:
        source_topk_freq_table = {k: v for k, v in source_freq_table.most_common(topk)}
        augmentation_topk_freq_table = {k: v for k, v in augmentation_freq_table.most_common(topk)}

        source_topk_freq, augmentation_topk_freq = [], []
        for cui in set(source_topk_freq_table.keys()) | set(augmentation_topk_freq_table.keys()):
            source_topk_freq.append(source_freq_table.get(cui, 1e-6) / len(sources))
            augmentation_topk_freq.append(augmentation_freq_table.get(cui, 1e-6) / len(augmentations))
        
        topk_kl_div = entropy(pk=source_topk_freq, qk=augmentation_topk_freq)
        print(f"Number of combine top-{topk} terms: {len(source_topk_freq)}")
        uni_result["statistics"][f"#combine top-{topk} terms"] = len(source_topk_freq)
        print(f">>>>>>>>>KL divergence of top-{topk} term frequency: {float(topk_kl_div)}<<<<<<<<<<<<<<<")
        uni_result["metrics"][f"top-{topk} KL divergence"] = topk_kl_div

    # compare doc frequency
    source_doc_cuis, augmentation_doc_cuis = [], []
    for doc_cuis in sources:
        for _, item in doc_cuis.items():
            source_doc_cuis += list(set([v[1] for v in item]))
    for doc_cuis in augmentations:
        for _, item in doc_cuis.items():
            augmentation_doc_cuis += list(set([v[1] for v in item]))
    
    source_doc_freq_table = Counter(source_doc_cuis)
    augmentation_doc_freq_table = Counter(augmentation_doc_cuis)

    source_doc_cuis_set = set(source_doc_freq_table.keys())
    augmentation_doc_cuis_set = set(augmentation_doc_freq_table.keys())
    # pdb.set_trace()
    doc_diff1 = source_doc_cuis_set - augmentation_doc_cuis_set
    doc_diff2 = augmentation_doc_cuis_set - source_doc_cuis_set
    doc_diff1_freq = {k: source_doc_freq_table[k] for k in doc_diff1}
    doc_diff1_freq_sorted = dict(sorted(doc_diff1_freq.items(), key=lambda item: item[1], reverse=True))
    doc_diff2_freq = {k: augmentation_doc_freq_table[k] for k in doc_diff2}
    doc_diff2_freq_sorted = dict(sorted(doc_diff2_freq.items(), key=lambda item: item[1], reverse=True))
    
    source_doc_freq, augmentation_doc_freq = [], []
    for cui in set(source_doc_freq_table.keys()) | set(augmentation_doc_freq_table.keys()):
        source_doc_freq.append(source_doc_freq_table.get(cui, 1e-6) / len(sources))
        augmentation_doc_freq.append(augmentation_doc_freq_table.get(cui, 1e-6) / len(augmentations))
    
    doc_kl_div = entropy(pk=source_doc_freq, qk=augmentation_doc_freq)
    uni_result["metrics"]["KL divergence of doc frequency"] = doc_kl_div
    print(f">>>>>>>>>KL divergence of doc frequency: {float(doc_kl_div)}<<<<<<<<<<<<<<<")

    # save result
    if output_folder:
        with open(os.path.join(output_folder, "uni_result.json"), "w") as f:
            json.dump(uni_result, f, indent=4)

    # get cumulative frequency
    compare_cumFreq(source_freq_table, augmentation_freq_table, T1label="original", T2label="synthetic", output_file=os.path.join(output_folder, "uni_histpgram.png") if output_folder else None)


def compare_bi_UMLS(sources, augmentations, topk=None, output_folder=None):
    print("==================================Results for bigram==================================")
    bi_result = {
        "statistics": {},
        "frequency difference": {},
        "metrics": {}
    }
    # prepare bigrams per section
    source_bigrams, augmentation_bigrams = [], []
    cui2idx = {}
    for doc_terms in sources:
        # for section_terms in doc_terms:
        for section in SECTIONS:
            if section not in doc_terms:
                continue
            section_terms = doc_terms[section]
            # get combination of all terms
            for i in range(len(section_terms)):
                for j in range(i+1, len(section_terms)):
                    term1, cui1 = section_terms[i]
                    term2, cui2 = section_terms[j]
                    if cui1 not in cui2idx:
                        cui2idx[cui1] = len(cui2idx)
                    if cui2 not in cui2idx:
                        cui2idx[cui2] = len(cui2idx)
                    if cui2idx[cui1] > cui2idx[cui2]:
                        source_bigrams.append((cui2idx[cui2], cui2idx[cui1]))
                    else:
                        source_bigrams.append((cui2idx[cui1], cui2idx[cui2]))
    
    for doc_terms in augmentations:
        # for section_terms in doc_terms:
        for section in SECTIONS:
            if section not in doc_terms:
                continue
            section_terms = doc_terms[section]
            # get combination of all terms
            for i in range(len(section_terms)):
                for j in range(i+1, len(section_terms)):
                    term1, cui1 = section_terms[i]
                    term2, cui2 = section_terms[j]
                    if cui1 not in cui2idx:
                        cui2idx[cui1] = len(cui2idx)
                    if cui2 not in cui2idx:
                        cui2idx[cui2] = len(cui2idx)
                    if cui2idx[cui1] > cui2idx[cui2]:
                        augmentation_bigrams.append((cui2idx[cui2], cui2idx[cui1]))
                    else:
                        augmentation_bigrams.append((cui2idx[cui1], cui2idx[cui2]))
    
    # get frequency table
    source_bigram_freq_table = Counter(source_bigrams)
    augmentation_bigram_freq_table = Counter(augmentation_bigrams)

    source_bigrams_set = set(source_bigram_freq_table.keys())
    augmentation_bigrams_set = set(augmentation_bigram_freq_table.keys())
    jac = jaccard_similarity(source_bigrams_set, augmentation_bigrams_set)
    print(f">>>>>>>>>Jaccard Similarity of bigrams: {float(jac)}<<<<<<<<<<<<<<<")
    bi_result["metrics"]["jaccard similarity"] = jac
    print(f"Number of unique bigrams in original corpus: {len(source_bigrams_set)}")
    bi_result["statistics"]["#source bigrams"] = len(source_bigrams_set)
    print(f"Number of unique bigrams in synthetic corpus: {len(augmentation_bigrams_set)}")
    bi_result["statistics"]["#synthetic bigrams"] = len(augmentation_bigrams_set)

    source_freq, augmentation_freq = [], []
    freq_diff = []
    for bigram in set(source_bigram_freq_table.keys()) | set(augmentation_bigram_freq_table.keys()):
        source_freq.append(source_bigram_freq_table.get(bigram, 1e-6) / len(sources))
        augmentation_freq.append(augmentation_bigram_freq_table.get(bigram, 1e-6) / len(augmentations))
        freq_diff.append(abs(source_bigram_freq_table.get(bigram, 0) - augmentation_bigram_freq_table.get(bigram, 0)))
    # print statistics for freq_diff
    print(f"Min-Max occurrence difference: {min(freq_diff)}-{max(freq_diff)}")
    bi_result["frequency difference"]["maximum"] = max(freq_diff)
    bi_result["frequency difference"]["minimum"] = min(freq_diff)
    print(f"Mean occurrence difference: {np.mean(freq_diff)}")
    bi_result["frequency difference"]["mean"] = np.mean(freq_diff)
    print(f"Median occurrence difference: {np.median(freq_diff)}")
    bi_result["frequency difference"]["median"] = np.median(freq_diff)
    print(f"p90 occurrence difference: {np.percentile(freq_diff, 90)}")
    bi_result["frequency difference"]["p90"] = np.percentile(freq_diff, 90)
    print(f"p95 occurrence difference: {np.percentile(freq_diff, 95)}")
    bi_result["frequency difference"]["p95"] = np.percentile(freq_diff, 95)
    print(f"p99 occurrence difference: {np.percentile(freq_diff, 99)}")
    bi_result["frequency difference"]["p99"] = np.percentile(freq_diff, 99)

    kl_div = entropy(pk=source_freq, qk=augmentation_freq)
    bi_result["metrics"]["KL divergence"] = kl_div
    print(f">>>>>>>>>KL divergence of bigram frequency: {float(kl_div)}<<<<<<<<<<<<<<<")
    if topk:
        source_topk_freq_table = {k: v for k, v in source_bigram_freq_table.most_common(topk)}
        augmentation_topk_freq_table = {k: v for k, v in augmentation_bigram_freq_table.most_common(topk)}

        source_topk_freq, augmentation_topk_freq = [], []
        for bigram in set(source_topk_freq_table.keys()) | set(augmentation_topk_freq_table.keys()):
            source_topk_freq.append(source_bigram_freq_table.get(bigram, 1e-6) / len(sources))
            augmentation_topk_freq.append(augmentation_bigram_freq_table.get(bigram, 1e-6) / len(augmentations))
        
        topk_kl_div = entropy(pk=source_topk_freq, qk=augmentation_topk_freq)
        print(f"Number of combine top-{topk} bigrams: {len(source_topk_freq)}")
        bi_result["statistics"][f"#combine top-{topk} bigrams"] = len(source_topk_freq)
        print(f">>>>>>>>>KL divergence of top-{topk} bigram frequency: {float(topk_kl_div)}<<<<<<<<<<<<<<<")
        bi_result["metrics"][f"top-{topk} KL divergence"] = topk_kl_div
    
    
    # save result
    if output_folder:
        with open(os.path.join(output_folder, "bi_result.json"), "w") as f:
            json.dump(bi_result, f, indent=4)

    # get cumulative frequency
    compare_cumFreq(source_bigram_freq_table, augmentation_bigram_freq_table, T1label="original", T2label="synthetic", output_file=os.path.join(output_folder, "bi_histpgram.png") if output_folder else None)


def calculate_note_section_coherence(note_terms):
    section_term_set = {}
    for section in SECTIONS:
        if section not in note_terms:
            continue
        section_term_set[section] = set([v[1] for v in note_terms[section]])
    # calculate coherence
    section_coherence = {}
    for i in range(len(SECTIONS)):
        if SECTIONS[i] not in note_terms or len(section_term_set[SECTIONS[i]]) == 0:
            continue
        section_coherence[SECTIONS[i]] = {}
        for j in range(i+1, len(SECTIONS)):
            if SECTIONS[j] not in note_terms:
                continue
            # calculate recall
            recall = len(section_term_set[SECTIONS[i]] & section_term_set[SECTIONS[j]]) / len(section_term_set[SECTIONS[i]])
            section_coherence[SECTIONS[i]][SECTIONS[j]] = recall
    return section_coherence

def section_coherence(sources, augmentations, output_folder=None):
    print("==================================Results for section coherence==================================")
    coherence_result = {}
    
    # calculate coherence for source notes
    source_coherence = {}
    for note_terms in sources:
        note_section_coherence = calculate_note_section_coherence(note_terms)
        for section in note_section_coherence:
            for next_section in note_section_coherence[section]:
                if section not in source_coherence:
                    source_coherence[section] = {}
                if next_section not in source_coherence[section]:
                    source_coherence[section][next_section] = []
                source_coherence[section][next_section].append(note_section_coherence[section][next_section])
    
    # calculate coherence for synthetic notes
    augmentation_coherence = {}
    for note_terms in augmentations:
        note_section_coherence = calculate_note_section_coherence(note_terms)
        for section in note_section_coherence:
            for next_section in note_section_coherence[section]:
                if section not in augmentation_coherence:
                    augmentation_coherence[section] = {}
                if next_section not in augmentation_coherence[section]:
                    augmentation_coherence[section][next_section] = []
                augmentation_coherence[section][next_section].append(note_section_coherence[section][next_section])
    
    
    # calculate average coherence
    source_coherence_avg = {}
    augmentation_coherence_avg = {}
    source_section_pair_score = []
    augmentation_section_pair_score = []
    for section in source_coherence:
        source_coherence_avg[section] = {}
        for next_section in source_coherence[section]:
            source_coherence_avg[section][next_section] = np.mean(source_coherence[section][next_section])
            source_section_pair_score.append(np.mean(source_coherence[section][next_section]))
    for section in augmentation_coherence:
        augmentation_coherence_avg[section] = {}
        for next_section in augmentation_coherence[section]:
            augmentation_coherence_avg[section][next_section] = np.mean(augmentation_coherence[section][next_section])
            augmentation_section_pair_score.append(np.mean(augmentation_coherence[section][next_section]))
    
    # calculate average coherence across section pairs
    source_average_coherence = np.mean(source_section_pair_score)
    augmentation_average_coherence = np.mean(augmentation_section_pair_score)
    
    print(f"Average coherence of source notes: {source_average_coherence}")
    print(f"Average coherence of synthetic notes: {augmentation_average_coherence}")

    
    coherence_result["average_coherence"] = {
        "original": source_average_coherence,
        "synthetic": augmentation_average_coherence
    }
    coherence_result["original"] = source_coherence_avg
    coherence_result["synthetic"] = augmentation_coherence_avg
    
    # plot heatmap
    source_matrix = np.full((len(SECTIONS), len(SECTIONS)), np.nan)
    augmentation_matrix = np.full((len(SECTIONS), len(SECTIONS)), np.nan)
    for i, sec_i in enumerate(SECTIONS):
        for j, sec_j in enumerate(SECTIONS):
            if i < j:
                if sec_i not in source_coherence_avg:
                    source_matrix[i, j] = np.nan
                else:
                    source_matrix[i, j] = source_coherence_avg[sec_i].get(sec_j, np.nan)
                if sec_i not in augmentation_coherence_avg:
                    augmentation_matrix[i, j] = np.nan
                else:
                    augmentation_matrix[i, j] = augmentation_coherence_avg[sec_i].get(sec_j, np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns
    sns.heatmap(source_matrix, annot=True, cmap='Blues', xticklabels=SECTIONS, yticklabels=SECTIONS,
                mask=np.isnan(source_matrix), ax=axes[0], square=True, fmt='.2f')
    axes[0].set_title("Coherence Heatmap for Source Notes")
    sns.heatmap(augmentation_matrix, annot=True, cmap='Blues', xticklabels=SECTIONS, yticklabels=SECTIONS,
                cbar_kws={'label': 'Coherence'},
                mask=np.isnan(augmentation_matrix), ax=axes[1], square=True, fmt='.2f')
    axes[1].set_title("Coherence Heatmap for Synthetic Notes")

    # save result
    if output_folder:
        with open(os.path.join(output_folder, "coherence_result.json"), "w") as f:
            json.dump(coherence_result, f, indent=4)
        # save source and augmentation coherence
        plt.savefig(os.path.join(output_folder, "coherence_heatmap.png"), dpi=300, bbox_inches='tight')

    # Display the plot
    plt.tight_layout()  # Adjust layout to make sure everything fits nicely
    plt.show()


if __name__ == "__main__":
    file = "Llama-3.2-1B-Instruct-SFT-DP-EP8/synthesized_data_dp_ep8_ppl_min.json"  # Update this path to your synthetic note file

    temp_folder = "temp/"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    umls_result_file = os.path.join(temp_folder, "umls_result.json")

    if file.endswith(".json"):
        dataset = json.load(open(file, "r"))
    elif file.endswith(".jsonl"):
        with open(file, "r") as f:
            dataset = [json.loads(line) for line in f]
    else:
        raise ValueError("Unsupported file format.")
    
    # call UMLS function
    # dataset = dataset[:10]
    source_UMLS_result, generation_UMLS_result = [], []
    _p = Pool(multiprocessing.cpu_count() - 4)
    for r in tqdm(_p.imap(extract_UMLS, dataset), total=len(dataset)):
        source_UMLS_result.append(r[0])
        generation_UMLS_result.append(r[1])
    _p.close()
    _p.join()

    # save UMLS results to temp file
    with open(umls_result_file, "w") as f:
        f.write(json.dumps({
            "source": source_UMLS_result,
            "synthetic": generation_UMLS_result
        }, indent=4))

    # call comparison function
    compare_uni_UMLS(source_UMLS_result, generation_UMLS_result, topk=1000, output_folder=temp_folder)
    compare_bi_UMLS(source_UMLS_result, generation_UMLS_result, topk=1000, output_folder=temp_folder)
    # section_coherence(source_UMLS_result, generation_UMLS_result, output_folder=temp_folder)

    
