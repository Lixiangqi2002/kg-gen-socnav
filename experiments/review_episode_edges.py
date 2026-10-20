"""Qwen source review of KGGen episode edges. Decisions remain model candidates."""

import argparse
import json
from pathlib import Path

import litellm

from socnav_six_run_pilot import EPISODES, save


SCHEMA = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "proposition_ids": {"type": "array", "items": {"type": "string"},
                            "maxItems": 4},
        "reason": {"type": "string"},
    },
    "required": ["supported", "proposition_ids", "reason"],
    "additionalProperties": False,
}


def review_edge(relation, propositions, output_path):
    valid_ids = {p["proposition_id"] for p in propositions}
    source = "\n".join(f"[{p['role'].upper()} {p['proposition_id']}] {p['text']}"
                       for p in propositions)
    prompt = (
        "Source propositions:\n" + source + "\n\nCandidate triple:\n" +
        json.dumps(relation, ensure_ascii=False) + "\n\n"
        "Is THIS directed subject-predicate-object claim supported by one or more "
        "propositions? Require the predicate and both endpoints to match the same "
        "claim, allowing clear coreference. A sentence saying two objects each remain "
        "still does not support a remain-still relation between them. Do not turn a "
        "one-place action into a relation to another entity. Coordinated subjects "
        "with a shared object do not relate to each other: 'A and B line C' does not "
        "support 'A lines B'. Separate clauses do not create a relation: 'A extends "
        "forward while B is open' does not support 'A extends forward B'. Preserve whether the "
        "source is observation, prediction, reasoning, or action. If unsupported or "
        "unclear, set supported=false and proposition_ids=[]."
    )
    response = litellm.completion(
        model="hosted_vllm/know2socnav-qwen3-30b-a3b", api_key="EMPTY",
        api_base="http://10.137.6.216:8000/v1", temperature=0,
        max_tokens=512,
        messages=[{"role": "system", "content": "Audit one KGGen candidate edge against source propositions. Return JSON only. /no_think"},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "edge_review", "strict": True, "schema": SCHEMA}},
    )
    choice = response.choices[0]
    save(output_path.with_suffix(".response.json"), {
        "finish_reason": choice.finish_reason, "raw": choice.message.content})
    if choice.finish_reason != "stop":
        raise ValueError(f"Incomplete response: {choice.finish_reason}")
    decision = json.loads(choice.message.content)
    ids = decision.get("proposition_ids", [])
    if any(item not in valid_ids for item in ids):
        raise ValueError(f"Unknown proposition ID in decision: {ids}")
    if decision.get("supported") and not ids:
        raise ValueError("Supported decision has no proposition IDs")
    decision["relation"] = relation
    save(output_path, decision)
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    source = json.loads((EPISODES / f"{args.episode_id}.json").read_text(encoding="utf-8"))
    propositions = source["call_0_normalization"]["propositions"]
    args.output.mkdir(parents=True, exist_ok=True)
    decisions = []
    for index, relation in enumerate(graph["relations"], 1):
        path = args.output / f"edge_{index:03d}.json"
        if path.exists():
            decision = json.loads(path.read_text(encoding="utf-8"))
        else:
            try:
                decision = review_edge(relation, propositions, path)
            except Exception as exc:
                save(args.output / f"edge_{index:03d}.error.json", {
                    "relation": relation, "error": repr(exc)})
                decision = {"relation": relation, "supported": False,
                            "proposition_ids": [], "reason": "review_failed"}
        decisions.append(decision)
        print(index, len(graph["relations"]), decision["supported"],
              decision["relation"], flush=True)
    save(args.output / "summary.json", {
        "episode_id": args.episode_id,
        "source_graph": str(args.graph.resolve()),
        "total": len(decisions),
        "model_supported": sum(bool(d["supported"]) for d in decisions),
        "decisions": decisions,
        "status": "Qwen review candidates; human review required",
    })


if __name__ == "__main__":
    main()
