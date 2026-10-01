"""
Script: Check whether canary placeholders leak in model generations.
Chat-format aware, vLLM-based, seed-based generation.

HOW IT WORKS
─────────────
For each (canary_type, seed) pair, the script runs your round-by-round
multi-turn inference loop:
  - All conversations (one per canary type here, but the loop is general)
    are processed round by round.
  - At each round, user turns are collected across all active conversations,
    batched, and sent to vLLM in one call.
  - The generated assistant response is appended to the conversation history
    and the loop continues with the next round — regardless of what the
    model said.
  - After all rounds, every generated assistant turn is scanned for verbatim
    occurrences of the canary placeholder.

REPORTING
──────────
  Seed-level   : per-seed leaked / not-leaked for each canary type.
  Aggregated   : all generated assistant turns pooled across ALL seeds;
                 leakage reported over this combined pool (total hits,
                 leak rate over all turns, leaked in any turn at all).

CANARY CONFIG FORMAT  (same file as perplexity_rank.py)
────────────────────
{
  "name": {
    "placeholder": "John Smith",
    "placeholder_marker": "<CANARY_NAME>",
    "conversation": [
      {"role": "system",    "content": "You are ..."},
      {"role": "user",      "content": "Generate a patient info section."},
      {"role": "assistant", "content": "Name: <CANARY_NAME> ..."},
      {"role": "user",      "content": "Now add the allergies section."},
      {"role": "assistant", "content": "Allergies: ..."}
    ]
  }
}

Only USER turns are used as prompts. ASSISTANT turns in the config are
reference only — the model generates its own.

Usage:
    python check_leakage.py \\
        --model_path /path/to/finetuned-llama-3.1-1b \\
        --canary_config canary_config.json \\
        --seeds 0 1 2 3 4 \\
        [--max_new_tokens 512] \\
        [--lora /path/to/lora/adapter] \\
        [--tensor_parallel 1] \\
        [--output_json leakage_results.json] \\
        [--output_generations generations.json] \\
        [--case_sensitive]

Dependencies:
    pip install vllm transformers tqdm
"""

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ---------------------------------------------------------------------------
# Round-by-round vLLM inference  (mirrors your run_evaluation_loop)
# ---------------------------------------------------------------------------

def run_conversation_loop(
    original_messages_list: list[list[dict]],
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None,
) -> list[list[str]]:
    """
    Run the round-by-round multi-turn generation loop over a list of
    conversations, using vLLM for batched inference.

    Args:
        original_messages_list : list of conversation templates, each a list
                                 of {"role": ..., "content": ...} dicts.
                                 Only user/system turns are used as prompts;
                                 assistant turns in the template are skipped.
        llm                    : vLLM LLM instance.
        tokenizer              : HuggingFace tokenizer (for apply_chat_template).
        sampling_params        : vLLM SamplingParams (seed set externally).
        lora_request           : LoRARequest or None.

    Returns:
        List of lists: for each conversation, the list of generated assistant
        turn strings (one per user turn).
    """
    # Initialise conversation state for each example
    all_conversations = []
    for idx, original_messages in enumerate(original_messages_list):
        all_conversations.append({
            "id":               idx,
            "messages":         [],          # growing history (user + generated assistant)
            "original_messages": original_messages,
            "generated_turns":  [],          # only the generated assistant texts
        })

    max_rounds        = max(len(c["original_messages"]) for c in all_conversations)
    active_indices    = list(range(len(all_conversations)))

    round_idx = 0
    while round_idx < max_rounds and active_indices:
        batch_inputs  = []
        batch_conv_ix = []

        for conv_idx in active_indices[:]:
            conv              = all_conversations[conv_idx]
            original_messages = conv["original_messages"]
            generated_messages = conv["messages"]

            if round_idx >= len(original_messages):
                active_indices.remove(conv_idx)
                continue

            current_message = original_messages[round_idx]

            # Add non-assistant turns to running history as-is
            if current_message["role"] != "assistant":
                generated_messages.append(current_message)

            if current_message["role"] == "user":
                # Format and queue for inference
                formatted_prompt = tokenizer.apply_chat_template(
                    generated_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                batch_inputs.append(formatted_prompt)
                batch_conv_ix.append(conv_idx)

            elif (round_idx + 1 < len(original_messages)
                  and original_messages[round_idx + 1]["role"] == "assistant"):
                # Next template turn is assistant → skip it (we generate our own)
                round_idx += 1

        if batch_inputs:
            outputs = llm.generate(
                prompts=batch_inputs,
                sampling_params=sampling_params,
                use_tqdm=False,
                lora_request=lora_request,
            )
            for j, output in enumerate(outputs):
                conv_idx      = batch_conv_ix[j]
                response_text = output.outputs[0].text.strip()

                all_conversations[conv_idx]["messages"].append(
                    {"role": "assistant", "content": response_text}
                )
                all_conversations[conv_idx]["generated_turns"].append(response_text)

        round_idx += 1

    return [c["generated_turns"] for c in all_conversations]


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------

def scan_turns(
    placeholder: str,
    turns: list[str],
    case_sensitive: bool,
) -> tuple[bool, int]:
    """Return (leaked, total_occurrences) across a list of turn strings."""
    needle = placeholder if case_sensitive else placeholder.lower()
    total  = sum(
        len(re.findall(re.escape(needle), (t if case_sensitive else t.lower())))
        for t in turns
    )
    return total > 0, total


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_seed_table(
    seed_results: dict[int, dict[str, bool]],
    canary_types: list[str],
) -> None:
    seeds = sorted(seed_results)
    col_w = 10
    header = f"| {'Seed':>6} | " + " | ".join(f"{ct:^{col_w}}" for ct in canary_types) + " |"
    sep    = "-" * len(header)
    print("\nSEED-LEVEL LEAKAGE  (✗ = leaked, ✓ = not leaked)")
    print(sep); print(header); print(sep)
    for seed in seeds:
        cells = [
            f"{'✗ YES':^{col_w}}" if seed_results[seed].get(ct, False)
            else f"{'✓ no':^{col_w}}"
            for ct in canary_types
        ]
        print(f"| {seed:>6} | " + " | ".join(cells) + " |")
    print(sep)


def print_aggregated_table(
    agg: dict[str, dict],
    canary_types: list[str],
) -> None:
    """
    One clinical note per seed (all assistant turns concatenated).
    Aggregated question: across N notes, how many contain the canary?
    """
    header = (
        f"| {'Type':<10} | {'Placeholder':<25} | "
        f"{'Total notes':>11} | {'Notes leaked':>12} | "
        f"{'Leak rate':>9} | {'Any leak?':>9} |"
    )
    sep = "-" * len(header)
    print("\nAGGREGATED LEAKAGE  (note-level, across all seeds)")
    print(sep); print(header); print(sep)

    for ct in canary_types:
        r = agg[ct]
        print(
            f"| {ct:<10} | {r['placeholder']:<25} | "
            f"{r['total_notes']:>11} | {r['notes_leaked']:>12} | "
            f"{r['leak_rate_pct']:>8.2f}% | "
            + (f"{'YES ✗':>9} |" if r["any_leaked"] else f"{'no  ✓':>9} |")
        )
    print(sep)

    n_leaked = sum(1 for r in agg.values() if r["any_leaked"])
    print(
        f"\n{n_leaked} / {len(canary_types)} canary types leaked in at least one note "
        f"({n_leaked / len(canary_types) * 100:.1f}%)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect canary leakage in chat-format vLLM generations (seed-based)."
    )
    parser.add_argument("--model_path",   type=str, required=True)
    parser.add_argument("--canary_config", type=str, default="canary_config.json")
    parser.add_argument("--canary_types", nargs="+", default=None)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(10)),
        help="Integer seeds for generation (default: 0–9). "
             "Each seed runs one full conversation per canary type.",
    )
    parser.add_argument("--max_new_tokens",  type=int,   default=2048)
    parser.add_argument("--lora",            type=str,   default=None,
                        help="Path to LoRA adapter directory (optional).")
    parser.add_argument("--tensor_parallel", type=int,   default=1)
    parser.add_argument("--output_json",     type=str,   default="leakage_results.json")
    parser.add_argument("--output_generations", type=str, default=None)
    parser.add_argument("--case_sensitive",  action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load canary config ────────────────────────────────────────────────────
    print(f"Loading canary config from: {args.canary_config}")
    with open(args.canary_config, "r", encoding="utf-8") as f:
        canary_config: dict[str, dict] = json.load(f)

    canary_types = args.canary_types or [k for k in canary_config if not k.startswith("_")]

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel,
        enable_lora=bool(args.lora),
        max_lora_rank=64 if args.lora else 16,
    )
    lora_request = LoRARequest("adapter", 1, args.lora) if args.lora else None
    tokenizer    = AutoTokenizer.from_pretrained(args.model_path)

    # ── Main loop: for each seed, run all canary conversations in one batch ───
    #
    # We batch ALL canary types together within a single vLLM call per round,
    # per seed — so vLLM sees the largest possible batch at every step.
    #
    # seed_results[seed][canary_type]          = leaked: bool
    # all_generations[canary_type][seed]       = [turn1, turn2, ...]
    # hit_counts[canary_type][seed]            = total occurrences

    seed_results:    dict[int, dict[str, bool]]       = {s: {} for s in args.seeds}
    all_generations: dict[str, dict[int, list[str]]]  = {ct: {} for ct in canary_types}
    hit_counts:      dict[str, dict[int, int]]         = {ct: {} for ct in canary_types}

    for seed in tqdm(args.seeds, desc="Seeds"):
        print(f"\n{'='*60}\nSeed {seed}")

        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id] + [128001, 128008],
            repetition_penalty=1.2,
            seed=seed,
        )

        # Build one conversation template per canary type for this seed.
        # We keep a mapping so we can attribute results back to canary types.
        ct_order = [ct for ct in canary_types if ct in canary_config]
        original_messages_list = [canary_config[ct]["conversation"] for ct in ct_order]

        # Run round-by-round loop over all canary conversations in parallel
        results_per_conv = run_conversation_loop(
            original_messages_list=original_messages_list,
            llm=llm,
            tokenizer=tokenizer,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        # Attribute results back to canary types
        for ct, generated_turns in zip(ct_order, results_per_conv):
            placeholder = canary_config[ct]["placeholder"]
            leaked, occurrences = scan_turns(
                placeholder, generated_turns, args.case_sensitive
            )
            seed_results[seed][ct]    = leaked
            all_generations[ct][seed] = generated_turns
            hit_counts[ct][seed]      = occurrences

            status = "✗ LEAKED" if leaked else "✓ clean"
            print(f"  [{ct}] seed {seed} → {status}  (hits: {occurrences})")

    # ── Aggregate: one note per seed (concatenate all assistant turns) ───────
    # A note leaks if the canary appears anywhere in the concatenated text.
    # Aggregated leak rate = notes_leaked / total_notes.
    aggregated: dict[str, dict] = {}
    for ct in canary_types:
        if ct not in canary_config:
            continue
        placeholder = canary_config[ct]["placeholder"]
        notes_leaked = 0
        for seed in args.seeds:
            # Concatenate all assistant turns for this seed → one clinical note
            note = "\n".join(all_generations[ct].get(seed, []))
            leaked, _ = scan_turns(placeholder, [note], args.case_sensitive)
            if leaked:
                notes_leaked += 1
        total_notes = len(args.seeds)
        aggregated[ct] = {
            "placeholder":   placeholder,
            "total_notes":   total_notes,
            "notes_leaked":  notes_leaked,
            "leak_rate_pct": round(notes_leaked / total_notes * 100, 2) if total_notes else 0.0,
            "any_leaked":    notes_leaked > 0,
        }

    # ── Print reports ─────────────────────────────────────────────────────────
    print_seed_table(seed_results, [ct for ct in canary_types if ct in canary_config])
    print_aggregated_table(aggregated, [ct for ct in canary_types if ct in canary_config])

    # ── Save results ──────────────────────────────────────────────────────────
    results_to_save = {
        "config": {
            "seeds":          args.seeds,
            "max_new_tokens": args.max_new_tokens,
            "case_sensitive": args.case_sensitive,
            "model_path":     args.model_path,
        },
        "seed_level": {
            ct: {
                "placeholder": canary_config[ct]["placeholder"],
                "per_seed": {
                    str(s): {"leaked": seed_results[s].get(ct, False)}
                    for s in args.seeds
                },
            }
            for ct in canary_types if ct in canary_config
        },
        "aggregated": aggregated,
    }

    out_path = Path(args.output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # ── Save generations (always) ─────────────────────────────────────────────
    # Structure: canary_type → seed → {leaked, note, turns}
    # "note"  = all assistant turns concatenated → the full clinical note.
    # "turns" = individual assistant responses, one per user turn, for traceability.
    gens_path = (
        Path(args.output_generations) if args.output_generations
        else out_path.with_name(out_path.stem + "_generations.json")
    )
    gens_to_save = {}
    for ct in canary_types:
        if ct not in canary_config:
            continue
        gens_to_save[ct] = {
            "placeholder": canary_config[ct]["placeholder"],
            "seeds": {
                str(seed): {
                    "leaked": seed_results[seed].get(ct, False),
                    "note":   "\n".join(all_generations[ct].get(seed, [])),
                    "turns":  all_generations[ct].get(seed, []),
                }
                for seed in args.seeds
            },
        }
    with open(gens_path, "w", encoding="utf-8") as f:
        json.dump(gens_to_save, f, indent=2)
    print(f"Generations saved to: {gens_path}")


if __name__ == "__main__":
    main()