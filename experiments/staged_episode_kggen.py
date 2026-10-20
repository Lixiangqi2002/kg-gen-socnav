"""Auditable whole-episode KGGen extraction with independent stage retries.

Nonempty output is not a quality gate. Every relation requires source review.
"""

import argparse
import hashlib
import json
from pathlib import Path

from kg_gen import KGGen
from kg_gen.models import Graph
from kg_gen.steps._1_get_entities import get_entities
from kg_gen.steps._2_get_relations import get_relations

from compare_episode_inputs import DEFAULT_IDS, inputs
from socnav_six_run_pilot import EPISODES, counts, graph_data, save


def capture(path):
    def hook(raw, finish_reason):
        save(path, {"finish_reason": finish_reason, "raw": raw})
    return hook


def rejected_relations(response_path, entities):
    if not response_path.exists():
        return []
    try:
        rows = json.loads(json.loads(response_path.read_text())["raw"])["relations"]
    except (ValueError, KeyError, TypeError):
        return []
    valid = set(entities)
    return [row for row in rows if isinstance(row, dict) and
            (row.get("subject") not in valid or row.get("object") not in valid)]


def structural_filter(rows):
    """Remove graph-form artifacts, not semantic source disagreements."""
    kept, rejected = set(), []
    for subject, predicate, obj in rows:
        reason = None
        if predicate.strip().lower() in {"is", "are", "role", "resource"}:
            reason = "bare_or_metadata_predicate"
        elif subject.startswith(("SNEI_EPISODE_", "MUSON_EPISODE_")) or \
                obj.startswith(("SNEI_EPISODE_", "MUSON_EPISODE_")):
            reason = "episode_id_endpoint"
        if reason:
            rejected.append({"relation": [subject, predicate, obj], "reason": reason})
        else:
            kept.add((subject, predicate, obj))
    return kept, rejected


def compact_normalized_input(episode_id):
    data = json.loads((EPISODES / f"{episode_id}.json").read_text(encoding="utf-8"))
    propositions = data["call_0_normalization"]["propositions"]
    return "\n\n".join(
        role.upper() + ": " + " ".join(p["text"] for p in propositions
                                      if p["role"] == role)
        for role in ("observation", "prediction", "reasoning", "action")
    )


def run_one(kg, text, folder, max_entity_attempts, max_relation_attempts,
            cluster=False, relaxed_last_attempt=False):
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    input_file = folder / "input.json"
    if input_file.exists() and json.loads(input_file.read_text())["sha256"] != digest:
        raise ValueError(f"Input changed for {folder}; use another output directory")
    save(input_file, {"text": text, "sha256": digest})

    entities_file = folder / "entities.json"
    if entities_file.exists():
        entities = json.loads(entities_file.read_text())["entities"]
    else:
        entities = []
        for attempt in range(1, max_entity_attempts + 1):
            stage = folder / f"entities_attempt_{attempt:02d}"
            try:
                entities = get_entities(
                    text, use_litellm_prompt=True, model=kg.model,
                    api_key=kg.api_key, api_base=kg.api_base,
                    temperature=kg.temperature, max_tokens=kg.max_tokens,
                    response_hook=capture(stage.with_suffix(".response.json")),
                )
                save(stage.with_suffix(".result.json"), {"count": len(entities)})
                if entities:
                    save(entities_file, {"entities": entities})
                    break
            except Exception as exc:
                save(stage.with_suffix(".error.json"), {"error": repr(exc)})
    if not entities:
        result = {"status": "entity_failed"}
        save(folder / "result.json", result)
        return result

    relations = set()
    for attempt in range(1, max_relation_attempts + 1):
        strict = not (relaxed_last_attempt and attempt == max_relation_attempts)
        stage = folder / f"relations_attempt_{attempt:02d}"
        response_path = stage.with_suffix(".response.json")
        try:
            found = get_relations(
                text, entities, use_litellm_prompt=True, model=kg.model,
                api_key=kg.api_key, api_base=kg.api_base,
                temperature=kg.temperature, max_tokens=kg.max_tokens,
                response_hook=capture(response_path), strict_entity_schema=strict,
            )
            rejected = rejected_relations(response_path, entities)
            filtered, structural_rejections = structural_filter(found)
            save(stage.with_suffix(".result.json"), {
                "strict_entity_schema": strict,
                "accepted_relations": [list(row) for row in sorted(filtered)],
                "rejected_endpoint_candidates": rejected,
                "structural_rejections": structural_rejections,
            })
            relations.update(filtered)
            if relations:
                break
        except Exception as exc:
            save(stage.with_suffix(".error.json"), {"error": repr(exc)})

    graph = Graph(entities=set(entities), relations=relations,
                  edges={row[1] for row in relations})
    save(folder / "raw_graph.json", graph_data(graph))
    result = {"status": "unreviewed_candidate" if relations else "zero_relations",
              "raw_counts": counts(graph)}
    if relations and cluster:
        try:
            clustered = kg.cluster(graph)
            save(folder / "clustered_graph.json", graph_data(clustered))
            result["clustered_counts"] = counts(clustered)
        except Exception as exc:
            result["cluster_error"] = repr(exc)
    save(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-ids", nargs="+", default=DEFAULT_IDS)
    parser.add_argument("--input-kind", choices=("original", "normalized",
                                                 "compact_normalized", "both"),
                        default="both")
    parser.add_argument("--entity-attempts", type=int, default=3)
    parser.add_argument("--relation-attempts", type=int, default=3)
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--relaxed-last-attempt", action="store_true",
                        help="Diagnostic only: allow free-text endpoints on the last try")
    args = parser.parse_args()
    if args.entity_attempts < 1 or args.relation_attempts < 1:
        parser.error("Attempt counts must be positive")
    kg = KGGen(model="hosted_vllm/know2socnav-qwen3-30b-a3b",
               api_key="EMPTY", api_base="http://10.137.6.216:8000/v1",
               max_tokens=4096, temperature=0.0,
               retrieval_model="sentence-transformers/all-MiniLM-L6-v2",
               disable_cache=True)
    for episode_id in args.episode_ids:
        variants = inputs(episode_id)
        variants["compact_normalized"] = compact_normalized_input(episode_id)
        for kind, text in variants.items():
            if args.input_kind not in ("both", kind):
                continue
            folder = args.output / episode_id / kind
            result = run_one(kg, text, folder, args.entity_attempts,
                             args.relation_attempts, cluster=args.cluster,
                             relaxed_last_attempt=args.relaxed_last_attempt)
            print(episode_id, kind, result, flush=True)


if __name__ == "__main__":
    main()
