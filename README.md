# Term2Note
Code for [Privacy-Preserving Generation of Clinical Narratives from Medical Terminologies](https://arxiv.org/abs/2509.10882) (Accepted to EMNLP 2026)

---

## Pre-requisites
Please have the following prepared before running the code:
1. [**UMLS Installation**](https://www.nlm.nih.gov/research/umls/implementation_resources/metamorphosys/help.html): We have the [2023AA-full version](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) installed.
2. [**SNOMED CT Map package**](https://www.nlm.nih.gov/healthit/snomedct/archive.html): We used the SNOMED CT International May 2023 version, i.e., `SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z`.
3. **Datasets**: 
    - Private training dataset $D_{train}^{src}$: [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/)
    - Private testing dataset $D_{test}^{src}$: [SNOMED CT Entity Linking Challenge dataset](https://physionet.org/content/snomed-ct-entity-challenge/1.0.0/)
    - Public dataset $D_{public}$: [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) or [PMC-Patients](https://huggingface.co/datasets/zhengyun21/PMC-Patients)
    <!-- - Other downstream task datasets: [Discharge Me](https://physionet.org/content/discharge-me/1.3/)  -->


<!-- --- -->
## Environment Setup
You will need two separate environments to run the code: one for FastDP and one for the rest of the code. The FastDP environment is used for differentially private training (and checkpoint conversion), while the other environment is used for data pre-processing, term generation, note generation inference, and evaluation.

1. `fastdp`: `python=3.8.20`, `pytorch=1.11.0`, `transformers==4.45.2`, `deepspeed==0.8.3`, `opacus==1.0.0`, `pydantic==1.10.8`, `fairscale==0.4.7`, `accelerate==0.26.0`, `datasets==3.1.0`, `ml-swissknife==0.1.30`
2. `term2note`: `python=3.12.9`, `torch==2.5.1`, `transformers==4.47.1`, `vllm==0.6.4.post1`, `bitsandbytes==0.45.2`, `sentence-transformers==3.4.1`, `accelerate==1.3.0`, `datasets==3.2.0`, `evaluate==0.4.4`, `mauve-text==0.4.0`, `scikit-learn==1.6.1`, `tiktoken==0.7.0`, `fire==0.7.0`, `seaborn`, [`quickumls`](https://github.com/Georgetown-IR-Lab/QuickUMLS)


<!-- --- -->
## Data pre-processing
Get the data ready for term generation and note generation by running **SecSplit** and **TermExt**. Unfortunately, we are not able to provide the processed data due to the data usage agreements of the pre-requisite datasets. However, we provide the code (`src/preprocessing`) to process the data. Please follow the steps below with environment `term2note` to prepare the data. Note that steps 2 and 3 need to be run for both the private datasets ($D_{train}^{src}$ and $D_{test}^{src}$) and $D_{public}$, while step 4 only needs to be run for the private datasets.

1. Retrieve MIMIC-IV/MIMIC-III notes from the source downloads. Please refer to `prepare_data/prepare_mimiciii_clean.py` and `prepare_data/prepare_mimiciv.py` in the repo [medical-coding-reproducibility](https://github.com/JoakimEdin/medical-coding-reproducibility) for MIMIC-III and MIMIC-IV notes, respectively. For MIMIC-IV, we only keep notes with ICD-10 codes by using the resulting `mimiciv_icd10` output, and manually exclude notes in the SNOMED CT Entity Linking Challenge dataset based on the `note_id`. Please also keep the `mimiciv_icd9` output, which is required by the utility evaluation.

2. **SecSplit**: to split each note into the six pre-defined sections. A template for the output is provided in `data/mimiv_iv_note_icd10_sections_template.json`.
    - run `src/preprocessing/extract_sections.py`; change lines 168 and 169 to the paths of the input file and the output file, respectively.


3. **TermExt**: to extract UMLS terms-of-interest from the notes. A template for the output is provided in `data/mimiv_iv_note_icd10_sections_umls_filtered_template.json`.
    - run `src/preprocessing/run_QuickUMLS.py` to get the UMLS terms for each note; update line 6 with your local path to the QuickUMLS installation; lines 53 and 55 are the input and output file paths, respectively.
    - run `src/preprocessing/cui2scui.py` to prepare the mapping between UMLS concept IDs (CUIs) and SNOMED CT concept IDs of the terms-of-interest; update lines 6-8 with your local paths to the required input and output files.
    - run `src/preprocessing/filter_UMLS.py` to filter out terms not in the CUI-to-SCUI mapping; update lines 4-9 with your local paths to the required input and output files.

    The output is a JSON file with each note containing the UMLS terms-of-interest.

4. Prepare data in conversation format for training and testing. Run `src/preprocessing/prepare_conversation.py` with the input file (output from step 3) and the output file paths provided as command line arguments, e.g.,
    ```bash
    python prepare_conversation.py --dataset PATH/TO/mimiciv_icd10_sections_umls_filtered.json --output_file PATH/TO/preprocessed_conversation_data_train.jsonl
    ```


<!-- --- -->
## Term Generation
Code is available in `src/TermGen`. Please follow the steps below with environment `term2note` to prepare the data and train the term generation model.

1. Prepare the training set: run `src/TermGen/prepare_terms.py` with the output of data pre-processing step 3 on $D_{public}$ to get the lists of terms for training; update lines 22-24 with your local paths to the required input and output files. A template for the output is provided in `data/mimiciii_terms_template.json`.

2. Fine-tune the LM to generate terms. Update the parameters in `sft_config.json` accordingly.
    ```bash
    # to prepare term embeddings first, run with do_train=false and do_eval=false to save embeddings
    CUDA_VISIBLE_DEVICES=0 python train.py --config sft_config.json
    ```

3. Inference: generate terms for each section with or without DP applied. Update the parameters in `inference_config.json` accordingly (`input_file_path` is the output of data pre-processing step 3 on $D_{test}^{src}$, and `generation_ckpt_path` is the `pytorch_model.bin` saved in step 2).
    ```bash
    CUDA_VISIBLE_DEVICES=0 python inference_full_pipeline.py --config inference_config.json
    ```
    Run `src/preprocessing/prepare_conversation.py` with the output file as input to prepare the generated terms in conversation format for note generation. 


<!-- --- -->
## Note Generation
Code is available in `src/NoteGen`.

1. Fine-tune with DP: run with environment `fastdp` to train the note generation model with differential privacy. The code is adapted from the repo [fast-differential-privacy](https://github.com/awslabs/fast-differential-privacy). Update `--train_data_file` in `run.sh` to the conversation-format training data (output of data pre-processing step 4 on $D_{train}^{src}$); note that 10% of the training file is held out for validation.
    ```bash
    bash run.sh
    ```

    Note that to fine-tune more recent models (e.g., Gemma), a newer version of Transformers is required to support the corresponding architecture. Therefore, it is recommended to have a separate environment with newer versions of Python, PyTorch, Transformers, etc., to avoid package conflicts. The package fastDP also needs to be adjusted accordingly to support the specific layers of the model. For example, for Gemma, uncomment the relevant lines in `fastDP/supported_layers_grad_samplers.py` and `fastDP/privacy_engine_dist_extending.py` by searching for the keyword `Gemma` in the files.


2. Convert the saved DeepSpeed checkpoint to safetensors format for inference with vLLM. Update `checkpoint_dir` and `tag` (lines 11-12) in `dp_convert.py` to match the checkpoint you want to convert, then run the following command in environment `fastdp`:
    ```bash
    python dp_convert.py
    ```

3. Inference: run with environment `term2note` to generate synthetic notes from the generated terms. `--model` is the directory where the converted model is saved in step 2, and `--dataset` is the conversation-format output of Term Generation step 3. Repeat with different seeds (we use 0, 42, 66 and 777) for the next step.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python infer_vllm_batch.py --model Llama-3.2-1B-Instruct-SFT-DP-EP8 --lora None --dataset PATH/TO/preprocessed_conversation_data_test.jsonl --output Llama-3.2-1B-Instruct-SFT-DP-EP8/synthesized_data_dp_ep8_seed0.json --seed 0
    ```

4. Apply DP maximiser: run with environment `term2note` to select, for each note, the synthetic note with the lowest perplexity among the generations from different seeds.
    ```bash
    CUDA_VISIBLE_DEVICES=0 python ppl.py --model /PATH/TO/Asclepius-Llama3-8B --predictions_file_path Llama-3.2-1B-Instruct-SFT-DP-EP8 --predictions_file_pattern synthesized_data_dp_ep8_seed --output_file synthesized_data_dp_ep8_ppl_min.json --batch_size 2 --max_length 8192
    ```

<!-- --- -->
## Evaluation
Code is available in `src/evaluation/`.

#### Fidelity Evaluation

 - `fidelity/analyse_datasets.py`: report the KL divergence of the note length distributions and the MAUVE score between synthetic notes and original notes (the MAUVE featurizer is set to `BioMistral-7B` on line 211)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python analyse_datasets.py --dataset1 "Llama-3.2-1B-Instruct-SFT-DP-EP8/synthesized_data_dp_ep8_ppl_min.json" --key1 original_messages --key2 messages --model Llama-3.2-1B-Instruct
    ```

 - `fidelity/compare_UMLS.py`: compare the UMLS term distributions in synthetic notes and original notes
    ```bash
    # change lines 26, 27 and 505 to your local paths to the required input files
    python compare_UMLS.py
    ```


#### Utility Evaluation - ICD coding prediction task
- `utility/asssign_ICD.py`: combine ICD-9 and ICD-10 chapter-level codes for notes in the SNOMED CT Entity Linking Challenge dataset, and split the dataset into 5 folds for cross-validation
    ```bash
    # change lines 142, 143, 146, 151, 211, 234 and 235 to your local paths to the required input and output files
    python asssign_ICD.py
    ```

- `utility/icd_classifier.py`: train and test the ICD coding prediction model on original notes and synthetic notes, respectively
    ```bash
    # train and test on original notes
    CUDA_VISIBLE_DEVICES=0 python icd_classifier.py --train_file "../../data/downstream/folds/train_comb_fold1.csv" --test_file "../../data/downstream/folds/test_comb_fold1.csv" --model_name yikuan8/Clinical-Longformer --num_epochs 30 --output_dir output_icd/original --fold 1

    # train on synthetic notes and test on original notes
    CUDA_VISIBLE_DEVICES=0 python icd_classifier.py --train_file "../../data/downstream/folds/train_comb_fold1.csv" --test_file "../../data/downstream/folds/test_comb_fold1.csv" --train_synth_file "Llama-3.2-1B-Instruct-SFT-DP-EP8/synthesized_data_dp_ep8_ppl_min.json" --model_name yikuan8/Clinical-Longformer --num_epochs 30 --output_dir output_icd/EP8 --fold 1
    ```
    Repeat for folds 1-5.

- `utility/aggregate_results.py`: aggregate results from 5-fold cross-validation
    ```bash
    # change lines 100 and 107 to your local paths
    python aggregate_results.py
    ```


#### Privacy Evaluation - Leakage of Private Information
For the privacy evaluation, the note generation model is fine-tuned (Note Generation step 1) on the training data with the canary conversations inserted. Due to the data usage agreement of MIMIC-IV, we only provide a masked template of the canary configuration in `privacy/pii_canary_config_template.json`: the clinical terms in the user messages and all note content except the canary placeholder in the assistant messages are masked. To reproduce the evaluation, create `pii_canary_config.json` from the template by filling in the masked parts with your own content (e.g., the "Patient Information" section of a training note and its clinical terms), keeping each `placeholder_marker` exactly once in the conversation. Use the same conversations for the canary-inserted training data and for the commands below.

 - `privacy/pii_canary_leakage.py`: check whether the canary placeholders are leaked in the synthetic notes
    ```bash
    # change model_path to your local path to the (DP) fine-tuned model on the canary-inserted dataset
    MKL_SERVICE_FORCE_INTEL=1 CUDA_VISIBLE_DEVICES=0 python pii_canary_leakage.py --model_path "Llama-3.2-1B-Instruct-PII-EP8" --canary_config pii_canary_config.json --seeds 0 42 66 777 --output_json pii_output/leakage_ppl_results_ep8.json --output_generations pii_output/ppl_generations_ep8.json
    ```

- `privacy/pii_perplexity_rank.py`: evaluate the perplexity rank of the true canary placeholder among all candidate placeholders under the fine-tuned model
    ```bash
    # change model_path to your local path to the (DP) fine-tuned model on the canary-inserted dataset
    # candidates_json is the JSON file containing all candidate canary placeholders for the perplexity rank (pii_all_candidates.json only shows the format)
    CUDA_VISIBLE_DEVICES=0 python pii_perplexity_rank.py --model_path "Llama-3.2-1B-Instruct-PII-EP8" --canary_config pii_canary_config.json --candidates_json pii_all_candidates.json --output_json pii_output/pii_perplexity_ranks_ep8.json
    ```
