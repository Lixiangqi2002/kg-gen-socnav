"""Retry failed KGGen episode candidates without losing the original attempt."""

import argparse
import json
import shutil
from pathlib import Path

from kg_gen import KGGen

from socnav_six_run_pilot import counts, graph_data, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    kg = KGGen(model="hosted_vllm/know2socnav-qwen3-30b-a3b",
               api_key="EMPTY", api_base="http://10.137.6.216:8000/v1",
               max_tokens=4096, temperature=0.0,
               retrieval_model="sentence-transformers/all-MiniLM-L6-v2",
               disable_cache=True)
    for folder in sorted((args.run / "episodes").iterdir()):
        if not folder.is_dir() or (folder / "clustered_graph.json").exists():
            continue
        attempts_dir = folder / "attempts"
        attempts_dir.mkdir(exist_ok=True)
        for name in ("raw_graph.json", "raw_counts.json", "cluster_error.json", "generation_error.json"):
            source = folder / name
            if source.exists() and not (attempts_dir / f"initial_{name}").exists():
                shutil.copy2(source, attempts_dir / f"initial_{name}")
        text = json.loads((folder / "input.json").read_text(encoding="utf-8"))["text"]
        for number in range(1, args.attempts + 1):
            prefix = attempts_dir / f"retry_{number:02d}"
            raw_path = prefix.with_name(prefix.name + "_raw_graph.json")
            error_path = prefix.with_name(prefix.name + "_error.json")
            if error_path.exists() and not raw_path.exists():
                continue
            try:
                if raw_path.exists():
                    from kg_gen.models import Graph
                    raw = Graph.from_file(str(raw_path))
                else:
                    raw = kg.generate(input_data=text, no_dspy=True,
                                      deduplication_method=None,
                                      context="One situated social navigation episode. Preserve observation, prediction, reasoning, and action distinctions.")
                    save(raw_path, graph_data(raw))
                save(prefix.with_name(prefix.name + "_counts.json"), counts(raw))
                if not raw.relations:
                    print(f"{folder.name} retry {number}: zero relations", flush=True)
                    continue
                clustered = kg.cluster(raw, context="One situated social navigation episode")
                save(prefix.with_name(prefix.name + "_clustered_graph.json"), graph_data(clustered))
                save(folder / "raw_graph.json", graph_data(raw))
                save(folder / "raw_counts.json", counts(raw))
                save(folder / "clustered_graph.json", graph_data(clustered))
                save(folder / "clustered_counts.json", counts(clustered))
                (folder / "cluster_error.json").unlink(missing_ok=True)
                (folder / "generation_error.json").unlink(missing_ok=True)
                save(folder / "selection.json", {"selected_attempt": number,
                                                 "previous_attempts": "attempts/"})
                print(f"{folder.name} retry {number}: complete {counts(clustered)}", flush=True)
                break
            except Exception as exc:
                save(error_path, {"error": repr(exc)})
                print(f"{folder.name} retry {number}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
