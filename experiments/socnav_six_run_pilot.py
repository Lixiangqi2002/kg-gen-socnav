"""Six KGGen pilot graphs from existing Know2SocNav evidence and episodes.

The evidence experiment extracts ten provenance-preserving batches and clusters
their aggregate once. Each episode is extracted and clustered independently.
Outputs are candidates for review, never validated knowledge.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from kg_gen import KGGen
from kg_gen.models import Graph


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "data/literature_kg_poc/runs/sentence_evidence"
EPISODES = ROOT / "data/episode_resources/runs/call01_freeze_final"
EPISODE_IDS = [
    "SNEI_EPISODE_00092c29d98b6d8de032",
    "SNEI_EPISODE_1d35713a4e8f4b3caa60",
    "SNEI_EPISODE_cee1b6ba9d824013e66a",
    "MUSON_EPISODE_330130dd576706e5e95f",
    "MUSON_EPISODE_4c7e6473d288d44bfe03",
]


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def graph_data(graph):
    return {
        "entities": sorted(graph.entities),
        "edges": sorted(graph.edges),
        "relations": sorted([list(x) for x in graph.relations]),
        "entity_clusters": {k: sorted(v) for k, v in (graph.entity_clusters or {}).items()},
        "edge_clusters": {k: sorted(v) for k, v in (graph.edge_clusters or {}).items()},
    }


def counts(graph):
    return {"entities": len(graph.entities), "edges": len(graph.edges),
            "relations": len(graph.relations),
            "entity_clusters": len(graph.entity_clusters or {}),
            "edge_clusters": len(graph.edge_clusters or {})}


def selected_evidence():
    by_paper = defaultdict(list)
    for path in sorted(EVIDENCE.glob("*/evidence_records/evidence_accepted.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "evidence_accepted" and record.get("evidence_text"):
                by_paper[record["provenance"]["paper_id"]].append(record)
    for records in by_paper.values():
        records.sort(key=lambda r: r["record_id"])
    chosen = []
    depth = 0
    while len(chosen) < 100:
        layer = [by_paper[p][depth] for p in sorted(by_paper) if len(by_paper[p]) > depth]
        if not layer:
            raise RuntimeError("Fewer than 100 accepted evidence records")
        chosen.extend(layer[:100 - len(chosen)])
        depth += 1
    return chosen


def extract_evidence_batch(kg, batch, folder):
    """Split only batches whose Qwen response cannot be parsed."""
    raw_path = folder / "raw_graph.json"
    if raw_path.exists():
        return Graph.from_file(str(raw_path))
    text = "\n\n".join(
        f"[Evidence {r['record_id']}; paper {r['provenance']['paper_id']}; "
        f"page {r['provenance']['page']}; chunk {r['provenance']['chunk_id']}]\n"
        f"{r['evidence_text']}" for r in batch)
    save(folder / "input.json", {"record_ids": [r["record_id"] for r in batch], "text": text})
    try:
        graph = kg.generate(input_data=text, no_dspy=True, deduplication_method=None,
                            context="Paper evidence in social robot navigation. Keep claims tied to evidence IDs.")
    except Exception as exc:
        save(folder / "generation_error.json", {"error": repr(exc)})
        if len(batch) == 1:
            raise
        middle = len(batch) // 2
        graph = kg.aggregate([
            extract_evidence_batch(kg, batch[:middle], folder / "split_a"),
            extract_evidence_batch(kg, batch[middle:], folder / "split_b"),
        ])
    save(raw_path, graph_data(graph))
    return graph


def run_graph(kg, text, folder, context):
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / "raw_graph.json"
    clustered_path = folder / "clustered_graph.json"
    if clustered_path.exists():
        return Graph.from_file(str(clustered_path)), "complete"
    if raw_path.exists():
        raw = Graph.from_file(str(raw_path))
    else:
        raw = kg.generate(input_data=text, context=context, no_dspy=True,
                          deduplication_method=None)
        save(raw_path, graph_data(raw))
    save(folder / "raw_counts.json", counts(raw))
    try:
        clustered = kg.cluster(raw, context=context)
    except Exception as exc:
        save(folder / "cluster_error.json", {"error": repr(exc)})
        return raw, "cluster_failed"
    save(clustered_path, graph_data(clustered))
    save(folder / "clustered_counts.json", counts(clustered))
    (folder / "cluster_error.json").unlink(missing_ok=True)
    return clustered, "complete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=["evidence", "episodes", "all"], default="all")
    args = parser.parse_args()
    output = args.output.resolve()
    save(output / "run_config.json", {
        "model": "hosted_vllm/know2socnav-qwen3-30b-a3b",
        "api_base": "http://10.137.6.216:8000/v1",
        "retrieval_model": "sentence-transformers/all-MiniLM-L6-v2",
        "temperature": 0.0,
        "extraction": "KGGen.generate(no_dspy=True, deduplication_method=None)",
        "clustering": "KGGen.cluster(default FULL; per episode or once for aggregate evidence)",
        "status": "candidate graphs for review",
    })
    kg = KGGen(model="hosted_vllm/know2socnav-qwen3-30b-a3b",
               api_key="EMPTY", api_base="http://10.137.6.216:8000/v1",
               max_tokens=4096, temperature=0.0,
               retrieval_model="sentence-transformers/all-MiniLM-L6-v2")

    if args.only in ("evidence", "all"):
        records = selected_evidence()
        base = output / "evidence_100"
        base.mkdir(parents=True, exist_ok=True)
        save(base / "sources.json", [{k: r[k] for k in ("record_id", "evidence_text", "sentence_ids", "provenance")}
                                     for r in records])
        batches = [records[i:i + 10] for i in range(0, 100, 10)]
        graphs = []
        for i, batch in enumerate(batches, 1):
            folder = base / f"batch_{i:02d}"
            graph = extract_evidence_batch(kg, batch, folder)
            graphs.append(graph)
            print(f"evidence batch {i}/10: {counts(graph)}", flush=True)
        aggregate = kg.aggregate(graphs)
        save(base / "raw_graph.json", graph_data(aggregate))
        save(base / "raw_counts.json", counts(aggregate))
        if not (base / "clustered_graph.json").exists():
            try:
                clustered = kg.cluster(aggregate, context="Paper evidence in social robot navigation")
                save(base / "clustered_graph.json", graph_data(clustered))
                save(base / "clustered_counts.json", counts(clustered))
                (base / "cluster_error.json").unlink(missing_ok=True)
                print(f"evidence clustered: {counts(clustered)}", flush=True)
            except Exception as exc:
                save(base / "cluster_error.json", {"error": repr(exc)})
                print(f"evidence clustering failed: {exc!r}", flush=True)

    if args.only in ("episodes", "all"):
        for i, episode_id in enumerate(EPISODE_IDS, 1):
            data = json.loads((EPISODES / f"{episode_id}.json").read_text(encoding="utf-8"))
            norm = data["call_0_normalization"]
            propositions = norm["propositions"]
            text = f"Episode {episode_id}. Resource {norm['resource']}.\n" + "\n".join(
                f"[{p['role'].upper()} {p['proposition_id']}] {p['text']}" for p in propositions)
            folder = output / "episodes" / episode_id
            save(folder / "input.json", {"episode_id": episode_id, "source_record_id": norm["source_record_id"],
                                         "propositions": propositions, "text": text})
            try:
                graph, status = run_graph(kg, text, folder,
                    "One situated social navigation episode. Preserve observation, prediction, reasoning, and action distinctions.")
                print(f"episode {i}/5 {episode_id}: {status} {counts(graph)}", flush=True)
            except Exception as exc:
                save(folder / "generation_error.json", {"error": repr(exc)})
                print(f"episode {i}/5 {episode_id}: generation_failed {exc!r}", flush=True)


if __name__ == "__main__":
    main()
