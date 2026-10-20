"""Five paper-local KGGen candidate graphs, extracted one Evidence Record at a time."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from kg_gen import KGGen
from kg_gen.steps._3_deduplicate import DeduplicateMethod

from socnav_plain_evidence import extract_plain_record
from socnav_six_run_pilot import EVIDENCE, counts, extract_evidence_record, graph_data, save


PAPER_IDS = ("SOC-REV-002", "SOC-REV-006", "SOC-REV-008", "SOC-REV-009", "SOC-REV-010")


def paper_records(paper_ids=PAPER_IDS):
    chosen = defaultdict(list)
    for path in sorted(EVIDENCE.glob("*/evidence_records/evidence_accepted.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            paper_id = record.get("provenance", {}).get("paper_id")
            if (paper_id in paper_ids and record.get("status") == "evidence_accepted"
                    and record.get("evidence_text")):
                chosen[paper_id].append(record)
    for paper_id in paper_ids:
        if not chosen[paper_id]:
            raise RuntimeError(f"No accepted evidence for {paper_id}")
        chosen[paper_id].sort(key=lambda record: record["record_id"])
    return chosen


def audit_graph(graph, records, folder):
    endpoints = {item for subject, _, obj in graph.relations for item in (subject, obj)}
    weak_matches = []
    pronoun_edges = []
    unresolved = {"he", "she", "it", "they", "them", "their", "this", "these", "those", "we", "us"}
    for subject, predicate, obj in sorted(graph.relations):
        if subject.casefold().strip() in unresolved or obj.casefold().strip() in unresolved:
            pronoun_edges.append([subject, predicate, obj])
        matching_ids = [record["record_id"] for record in records
                        if subject.casefold() in record["evidence_text"].casefold()
                        and obj.casefold() in record["evidence_text"].casefold()]
        if not matching_ids:
            weak_matches.append({"relation": [subject, predicate, obj],
                                 "reason": "No single evidence sentence contains both endpoint strings; review paraphrases manually"})
    save(folder / "review_flags.json", {
        "status": "candidate_requires_source_review",
        "no_entities": not bool(graph.entities),
        "no_relations": not bool(graph.relations),
        "isolated_entities": sorted(graph.entities - endpoints),
        "relations_needing_source_check": weak_matches,
        "pronoun_endpoint_relations": pronoun_edges,
    })


def run_recordwise(kg, paper_id, records, output, input_mode="plain", skip_clustering=False):
    suffix = "plain_recordwise" if input_mode == "plain" else "recordwise"
    folder = output / "papers" / f"{paper_id}_{suffix}"
    sources = [{key: r[key] for key in ("record_id", "evidence_text", "sentence_ids", "provenance")}
               for r in records]
    source_file = folder / "sources.json"
    if source_file.exists() and json.loads(source_file.read_text(encoding="utf-8")) != sources:
        raise RuntimeError(f"Recordwise sources changed for {paper_id}; use a fresh output directory")
    save(source_file, sources)
    graphs = []
    failed = []
    for i, record in enumerate(records, 1):
        try:
            extract = extract_plain_record if input_mode == "plain" else extract_evidence_record
            graph = extract(kg, record, folder / "records" / record["record_id"])
            graphs.append(graph)
            print(f"{paper_id} record {i}/{len(records)}: {counts(graph)}", flush=True)
        except Exception as exc:
            failed.append({"record_id": record["record_id"], "error": repr(exc)})
            print(f"{paper_id} record {i}/{len(records)}: failed {exc!r}", flush=True)
    save(folder / "record_failures.json", failed)
    audit_missing = [record["record_id"] for record in records
                     if input_mode == "plain" and
                     not (folder / "records" / record["record_id"] / "source_audit.json").exists()]
    save(folder / "paper_status.json", {
        "paper_id": paper_id, "source_records": len(records), "extracted_records": len(graphs),
        "failed_records": failed, "source_audits_missing": audit_missing,
        "status": "complete" if not failed and not audit_missing else "needs_attention",
    })
    if not graphs:
        return
    raw = kg.aggregate(graphs)
    save(folder / "raw_graph.json", graph_data(raw))
    save(folder / "raw_counts.json", counts(raw))
    audit_graph(raw, records, folder)
    if skip_clustering:
        return
    if not raw.relations:
        save(folder / "cluster_error.json", {"error": "No relations to cluster"})
        return
    try:
        clustered = kg.deduplicate(raw, method=DeduplicateMethod.SEMHASH)
        save(folder / "clustered_graph.json", graph_data(clustered))
        save(folder / "clustered_counts.json", counts(clustered))
        save(folder / "clustering_method.json", {"method": "KGGen SEMHASH", "similarity_threshold": 0.95})
        (folder / "cluster_error.json").unlink(missing_ok=True)
    except Exception as exc:
        save(folder / "cluster_error.json", {"error": repr(exc)})
        print(f"{paper_id} recordwise cluster_failed {exc!r}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--paper-ids", nargs="+", choices=PAPER_IDS, default=list(PAPER_IDS))
    parser.add_argument("--input-mode", choices=("plain", "original"), default="plain")
    parser.add_argument("--skip-clustering", action="store_true",
                        help="Save raw graphs first; avoid blocking on LLM deduplication")
    args = parser.parse_args()
    paper_ids = tuple(dict.fromkeys(args.paper_ids))
    records_by_paper = paper_records(paper_ids)
    output = args.output.resolve()
    save(output / "run_config.json", {
        "paper_ids": paper_ids,
        "extraction_unit": "one Evidence Record per call, aggregated within each paper",
        "input_mode": args.input_mode,
        "clustering": "KGGen SEMHASH separately for each paper",
        "review": "Plain rephrase and every edge get a Qwen source check; human review remains required",
        "model": "hosted_vllm/know2socnav-qwen3-30b-a3b",
        "status": "candidate graphs for source review",
    })
    kg = KGGen(model="hosted_vllm/know2socnav-qwen3-30b-a3b", api_key="EMPTY",
               api_base="http://10.137.6.216:8000/v1",
               max_tokens=1024 if args.input_mode == "plain" else 4096,
               temperature=0.0, retrieval_model="sentence-transformers/all-MiniLM-L6-v2")
    for paper_id in paper_ids:
        records = records_by_paper[paper_id]
        run_recordwise(kg, paper_id, records, output, args.input_mode, args.skip_clustering)


if __name__ == "__main__":
    main()
