"""Compare KGGen extraction from original fields and normalized propositions.

Each attempt is independent. Graphs remain unreviewed candidates.
"""

import argparse
import hashlib
import json
from pathlib import Path

from kg_gen import KGGen

from socnav_six_run_pilot import EPISODES, counts, graph_data, save


DEFAULT_IDS = (
    "MUSON_EPISODE_330130dd576706e5e95f",
    "MUSON_EPISODE_4c7e6473d288d44bfe03",
)


def inputs(episode_id):
    data = json.loads((EPISODES / f"{episode_id}.json").read_text(encoding="utf-8"))
    norm = data["call_0_normalization"]
    propositions = norm["propositions"]
    source = {}
    for p in propositions:
        source.setdefault((p["source_part_id"], p["source_field"]), p["source_text"])
    original = f"Episode {episode_id}. Resource {norm['resource']}.\n" + "\n".join(
        f"[{field}] {text}" for (_, field), text in source.items()
    )
    normalized = f"Episode {episode_id}. Resource {norm['resource']}.\n" + "\n".join(
        f"[{p['role'].upper()} {p['proposition_id']}] {p['text']}" for p in propositions
    )
    return {"original": original, "normalized": normalized}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-ids", nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cluster", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    kg = KGGen(model="hosted_vllm/know2socnav-qwen3-30b-a3b",
               api_key="EMPTY", api_base="http://10.137.6.216:8000/v1",
               max_tokens=4096, temperature=0.0,
               retrieval_model="sentence-transformers/all-MiniLM-L6-v2",
               disable_cache=True)
    for episode_id in args.episode_ids:
        for input_kind, text in inputs(episode_id).items():
            base = args.output / episode_id / input_kind
            save(base / "input.json", {
                "episode_id": episode_id,
                "input_kind": input_kind,
                "text": text,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            })
            for repeat in range(1, args.repeats + 1):
                folder = base / f"repeat_{repeat:02d}"
                if (folder / "result.json").exists():
                    continue
                try:
                    raw = kg.generate(input_data=text, no_dspy=True,
                                      deduplication_method=None)
                    save(folder / "raw_graph.json", graph_data(raw))
                    result = {"status": "extracted", "raw_counts": counts(raw)}
                    if args.cluster and raw.relations:
                        try:
                            clustered = kg.cluster(raw)
                            save(folder / "clustered_graph.json", graph_data(clustered))
                            result["clustered_counts"] = counts(clustered)
                        except Exception as exc:
                            result["cluster_error"] = repr(exc)
                except Exception as exc:
                    result = {"status": "extraction_failed", "error": repr(exc)}
                save(folder / "result.json", result)
                print(episode_id, input_kind, repeat, result, flush=True)


if __name__ == "__main__":
    main()
