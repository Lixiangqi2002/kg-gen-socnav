"""Run whole-episode normalized text through KGGen's original DSPy path.

One episode produces one unreviewed situated graph candidate. Proposition IDs and
source fields are saved alongside the compact model input for later edge review.
"""

import argparse
import json
from pathlib import Path

from kg_gen import KGGen

from socnav_six_run_pilot import EPISODES, EPISODE_IDS, counts, graph_data, save


def compact_text(propositions):
    return "\n\n".join(
        role.upper() + ": " + " ".join(p["text"] for p in propositions
                                      if p["role"] == role)
        for role in ("observation", "prediction", "reasoning", "action")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-ids", nargs="+", default=EPISODE_IDS)
    parser.add_argument("--cluster", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    save(output / "run_config.json", {
        "extraction": "KGGen.generate(no_dspy=False, deduplication_method=None)",
        "model": "hosted_vllm/know2socnav-qwen3-30b-a3b",
        "input": "one complete episode, normalized propositions as four role paragraphs",
        "provenance": "proposition IDs and exact source fields retained in each input.json",
        "clustering": args.cluster,
        "status": "unreviewed candidates",
    })
    kg = KGGen(model="hosted_vllm/know2socnav-qwen3-30b-a3b",
               api_key="EMPTY", api_base="http://10.137.6.216:8000/v1",
               max_tokens=4096, temperature=0.0,
               retrieval_model="sentence-transformers/all-MiniLM-L6-v2",
               disable_cache=True)
    for episode_id in args.episode_ids:
        source = json.loads((EPISODES / f"{episode_id}.json").read_text(encoding="utf-8"))
        norm = source["call_0_normalization"]
        folder = output / "episodes" / episode_id
        text = compact_text(norm["propositions"])
        input_file = folder / "input.json"
        saved = {"episode_id": episode_id,
                 "source_record_id": norm["source_record_id"],
                 "propositions": norm["propositions"], "text": text}
        if input_file.exists() and json.loads(input_file.read_text(encoding="utf-8")) != saved:
            raise ValueError(f"Input changed for {episode_id}; use a new run directory")
        save(input_file, saved)
        raw_file = folder / "raw_graph.json"
        if raw_file.exists():
            print(episode_id, "already extracted", flush=True)
            continue
        try:
            graph = kg.generate(input_data=text, no_dspy=False,
                                deduplication_method=None)
            save(raw_file, graph_data(graph))
            save(folder / "raw_counts.json", counts(graph))
            print(episode_id, "raw", counts(graph), flush=True)
            if args.cluster and graph.relations:
                clustered = kg.cluster(graph)
                save(folder / "clustered_graph.json", graph_data(clustered))
                save(folder / "clustered_counts.json", counts(clustered))
                print(episode_id, "clustered", counts(clustered), flush=True)
        except Exception as exc:
            save(folder / "generation_error.json", {"error": repr(exc)})
            print(episode_id, "failed", repr(exc), flush=True)


if __name__ == "__main__":
    main()
