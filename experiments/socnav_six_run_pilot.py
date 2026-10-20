"""Six KGGen pilot graphs from existing Know2SocNav evidence and episodes.

The evidence experiment extracts each source record independently, then clusters
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
    # KGGen clustering can return relation endpoints absent from its entity set.
    # Keep the saved graph internally complete without inventing relations.
    for subject, predicate, obj in graph.relations:
        graph.entities.update((subject, obj))
        graph.edges.add(predicate)
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


def selected_evidence(source_path=None):
    if source_path is not None:
        chosen = json.loads(Path(source_path).read_text(encoding="utf-8"))
        if len(chosen) != 100 or len({r["record_id"] for r in chosen}) != 100:
            raise ValueError("The evidence source manifest must contain 100 distinct records")
        return chosen
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


def save_evidence_review_flags(graph, record_id, folder):
    endpoints = {item for subject, _, obj in graph.relations for item in (subject, obj)}
    save(folder / "review_flags.json", {
        "record_id": record_id,
        "no_entities": not bool(graph.entities),
        "no_relations": not bool(graph.relations),
        "isolated_entities": sorted(graph.entities - endpoints),
        "status": "candidate_requires_source_review",
    })


def extract_evidence_record(kg, record, folder):
    """Keep extraction local to one Evidence Record and its source sentence."""
    raw_path = folder / "raw_graph.json"
    if raw_path.exists():
        graph = Graph.from_file(str(raw_path))
        save_evidence_review_flags(graph, record["record_id"], folder)
        return graph
    save(folder / "input.json", {k: record[k] for k in
                                ("record_id", "evidence_text", "sentence_ids", "provenance")})
    errors = []
    for _ in range(3):
        try:
            graph = kg.generate(input_data=record["evidence_text"], no_dspy=True,
                                deduplication_method=None)
            break
        except Exception as exc:
            errors.append(repr(exc))
    else:
        save(folder / "generation_error.json", {"attempts": errors})
        raise RuntimeError(f"Extraction failed for {record['record_id']}: {errors[-1]}")
    (folder / "generation_error.json").unlink(missing_ok=True)
    save(raw_path, graph_data(graph))
    save(folder / "raw_counts.json", counts(graph))
    save_evidence_review_flags(graph, record["record_id"], folder)
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
    parser.add_argument("--evidence-sources", type=Path,
                        help="Frozen sources.json with the exact 100 Evidence Records to replay")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.only in ("evidence", "all"):
        legacy = output / "evidence_100"
        if (legacy / "raw_graph.json").exists() and not (legacy / "records").exists():
            raise RuntimeError("This output has legacy multi-record batches; use a fresh output directory")
    save(output / "run_config.json", {
        "model": "hosted_vllm/know2socnav-qwen3-30b-a3b",
        "api_base": "http://10.137.6.216:8000/v1",
        "retrieval_model": "sentence-transformers/all-MiniLM-L6-v2",
        "temperature": 0.0,
        "extraction": "KGGen.generate(no_dspy=True, deduplication_method=None)",
        "evidence_extraction_unit": "one accepted Evidence Record at a time",
        "evidence_sources": str(args.evidence_sources.resolve()) if args.evidence_sources else "current accepted records",
        "clustering": "KGGen.cluster(default FULL; per episode or once for aggregate evidence)",
        "status": "candidate graphs for review",
    })
    kg = KGGen(model="hosted_vllm/know2socnav-qwen3-30b-a3b",
               api_key="EMPTY", api_base="http://10.137.6.216:8000/v1",
               max_tokens=4096, temperature=0.0,
               retrieval_model="sentence-transformers/all-MiniLM-L6-v2")

    if args.only in ("evidence", "all"):
        records = selected_evidence(args.evidence_sources)
        base = output / "evidence_100"
        base.mkdir(parents=True, exist_ok=True)
        sources = [{k: r[k] for k in ("record_id", "evidence_text", "sentence_ids", "provenance")}
                   for r in records]
        source_file = base / "sources.json"
        if (base / "records").exists() and source_file.exists():
            if json.loads(source_file.read_text(encoding="utf-8")) != sources:
                raise RuntimeError("Existing record outputs use different Evidence sources; use a fresh output directory")
        save(source_file, sources)
        graphs = []
        for i, record in enumerate(records, 1):
            folder = base / "records" / record["record_id"]
            graph = extract_evidence_record(kg, record, folder)
            graphs.append(graph)
            print(f"evidence {i}/100 {record['record_id']}: {counts(graph)}", flush=True)
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
