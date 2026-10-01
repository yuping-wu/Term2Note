import os, multiprocessing, csv, tqdm, ast, json
import pandas as pd
from multiprocessing.pool import Pool
from quickumls import QuickUMLS

QUICKUMLS_PATH = "/PATH/TO/QUICKUMLS/DATA" # Update this path to your QuickUMLS data directory

def match(params):
    # define QuickUMLS: using default parameters
    matcher = QuickUMLS(QUICKUMLS_PATH, threshold=0.5)
    nid, text = params
    umls_result = matcher.match(text, best_match=True, ignore_syntax=False)

     # format UMLS results
    new_umls_result = {} # "term": {sidx, eidx, [term, cui, similarity]}
    for result in umls_result:
        ngram = result[0]["ngram"]
        sidx, eidx = result[0]["start"], result[0]["end"]
        term_cui_sim_list = [[result[0]["term"], result[0]["cui"], result[0]["similarity"]]]

        for r in result[1:]:
            try:
                assert ngram == r["ngram"]
                assert sidx == r["start"]
                assert eidx == r["end"]
            except:
                print(f"Unmatched ngram/indices error for note {nid} with ngram {ngram}")
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
        print(f"Unmatched number of ngrams error for note {nid}")
    
    return {
        "note_id": nid,
        "text": text,
        "UMLS": [[k, v] for k, v in sorted_umls_result.items()]
    }


if __name__ == '__main__':
    # load dataset
    datasets = ["../../data/path-to-mimiciv_icd10.csv"]
    # note_id, text
    output_files = ["../../data/path-to-mimiciv_icd10_umls.jsonl"]
    
    # iterate over dataset
    for dataset, outfile in zip(datasets, output_files):
        fname = dataset[:-4]
        print('-'*20, 'Processing ', dataset, '-'*20,'\n')
        todo = []
        df = pd.read_csv(dataset)
        for i in range(len(df)):
            todo.append((df.loc[i, 'note_id'], df.loc[i, 'text']))

        _p = Pool(multiprocessing.cpu_count()-8)
        for r in tqdm.tqdm(_p.imap(match, todo), total=len(todo)):
            with open(outfile, "a") as f:
                f.write(json.dumps(r, default=list) + "\n")
        
        _p.close()
        _p.join()

        print("Done with identifying UMLS terms.")
        