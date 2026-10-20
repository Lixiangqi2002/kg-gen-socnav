"""Source-preserving plain rephrase and zero-edge repair for Evidence Records.

Every output is a candidate. The original Evidence sentence remains the review
authority; neither a rephrase nor a repaired edge is promoted automatically.
"""

import json

import litellm
from kg_gen.models import Graph

from socnav_six_run_pilot import counts, graph_data, save


MODEL = "hosted_vllm/know2socnav-qwen3-30b-a3b"
API_BASE = "http://10.137.6.216:8000/v1"


def completion(system, user, schema, name, max_tokens=1024):
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "system", "content": system + " Return only JSON. /no_think"},
                  {"role": "user", "content": user}],
        temperature=0.0, max_tokens=max_tokens,
        timeout=45,
        response_format={"type": "json_schema", "json_schema": {
            "name": name, "strict": True, "schema": schema}},
        api_key="EMPTY", api_base=API_BASE,
    )
    choice = response.choices[0]
    raw = choice.message.content or ""
    if choice.finish_reason == "length":
        raise ValueError(f"{name} response was truncated")
    return json.loads(raw), raw


PLAIN_SCHEMA = {"type": "object", "properties": {"plain_text": {"type": "string"}},
                "required": ["plain_text"], "additionalProperties": False}
ENTITIES_SCHEMA = {"type": "object", "properties": {
    "entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"], "additionalProperties": False}
RELATION_ITEM = {"type": "object", "properties": {
    "subject": {"type": "string"}, "predicate": {"type": "string"}, "object": {"type": "string"}},
    "required": ["subject", "predicate", "object"], "additionalProperties": False}
RELATIONS_SCHEMA = {"type": "object", "properties": {
    "relations": {"type": "array", "items": RELATION_ITEM}},
    "required": ["relations"], "additionalProperties": False}
EDGE_REVIEW = {"type": "object", "properties": {
    "index": {"type": "integer"},
    "verdict": {"type": "string", "enum": ["supported", "unsupported", "unclear"]},
    "reason": {"type": "string"}},
    "required": ["index", "verdict", "reason"], "additionalProperties": False}
SOURCE_AUDIT_SCHEMA = {"type": "object", "properties": {
    "rephrase_faithful": {"type": "boolean"},
    "rephrase_issues": {"type": "array", "items": {"type": "string"}},
    "edge_reviews": {"type": "array", "items": EDGE_REVIEW}},
    "required": ["rephrase_faithful", "rephrase_issues", "edge_reviews"],
    "additionalProperties": False}


def plain_rephrase(source, folder):
    path = folder / "plain_rephrase.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["plain_text"]
    system = ("Rewrite one paper Evidence sentence as short, plain English sentences. "
              "Preserve every assertion, actor, negation, uncertainty, comparison, and condition. "
              "Keep qualifiers such as potential, certain, socially aware, and may. "
              "Keep example-of, part-of, and definition relationships exactly as stated. "
              "Do not turn an illustrative example into a general claim. "
              "Keep technical terms. Resolve a pronoun only if its referent is explicit in this sentence. "
              "Do not add facts or strengthen a claim. Return a plain_text string.")
    for attempt in range(2):
        try:
            parsed, raw = completion(system, source, PLAIN_SCHEMA, "plain_rephrase")
            plain = parsed["plain_text"].strip()
            if not plain:
                raise ValueError("Empty plain rephrase")
            save(path, {"plain_text": plain, "source_text": source,
                        "raw_response": raw, "review_status": "needs_source_check"})
            return plain
        except Exception as exc:
            save(folder / "rephrase_error.json", {"attempt": attempt + 1, "error": repr(exc)})
    raise RuntimeError("Plain rephrase failed twice")


def repair_zero_edges(plain, initial, folder):
    """Expand missed source phrases, then retry relations without forcing an edge."""
    entity_system = ("Extract exact phrases that could be subjects or objects of explicit claims. "
                     "Include abstract concepts, properties, states, and processes, not just people. "
                     "For X is Y, include X and Y. Do not invent phrases. Return an entities array.")
    parsed, raw = completion(entity_system, plain, ENTITIES_SCHEMA, "evidence_entities")
    proposed = [e.strip() for e in parsed["entities"] if isinstance(e, str) and e.strip()]
    grounded = {e for e in proposed if e.casefold() in plain.casefold()}
    entities = sorted(set(initial.entities) | grounded)
    save(folder / "repair_entities.json", {"proposed": proposed, "grounded": sorted(grounded),
                                           "all_entities": entities, "raw_response": raw})
    if len(entities) < 2:
        return initial
    relation_system = ("Extract every subject-predicate-object relation explicitly supported by the text. "
                       "Subject and object must exactly match one supplied entity. "
                       "Preserve negation and modality in the predicate. "
                       "Do not create a triple for a one-place action or invent a link. "
                       "An empty relations array is allowed when no valid binary relation exists.")
    user = ("<entities>\n" + "\n".join(f"- {e}" for e in entities) +
            "\n</entities>\n<text>\n" + plain + "\n</text>")
    attempts = []
    for _ in range(2):
        try:
            parsed, raw = completion(relation_system, user, RELATIONS_SCHEMA,
                                     "evidence_relations", max_tokens=1024)
            relations = []
            added_endpoints = set()
            for item in parsed["relations"]:
                triple = (item["subject"], item["predicate"], item["object"])
                endpoints = (triple[0], triple[2])
                if not all(e in entities or e.casefold() in plain.casefold() for e in endpoints):
                    continue
                if triple[0] != triple[2] and triple[1].strip():
                    relations.append(triple)
                    added_endpoints.update(e for e in endpoints if e not in entities)
            attempts.append({"raw_response": raw, "accepted_source_phrase_relations": relations,
                             "added_endpoint_phrases": sorted(added_endpoints)})
            if relations:
                save(folder / "repair_attempts.json", attempts)
                return Graph(entities=set(entities) | added_endpoints, relations=set(relations),
                             edges={r[1] for r in relations})
        except Exception as exc:
            attempts.append({"error": repr(exc)})
    save(folder / "repair_attempts.json", attempts)
    return initial


def audit_against_original(source, plain, graph, folder):
    path = folder / "source_audit.json"
    relations = sorted(graph.relations)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if {r["index"] for r in existing["edge_reviews"]} == set(range(len(relations))):
            return
        save(folder / "source_audit_invalid.json", existing)
        path.unlink()
    if not relations:
        save(path, {"rephrase_faithful": None, "rephrase_issues": [],
                    "edge_reviews": [], "relations": [], "status": "no_edges_to_audit"})
        (folder / "source_audit_error.json").unlink(missing_ok=True)
        return
    system = ("Audit a plain rephrase and candidate graph against the ORIGINAL Evidence sentence only. "
              "Check whether the rephrase preserves qualifiers, negation, uncertainty, actors, "
              "and example versus type relationships. For each numbered triple, mark supported "
              "only if the original sentence asserts that exact relation with the same scope. "
              "Mark unsupported for changed or invented relations; mark unclear when context is missing. "
              "Review every supplied index. Give a brief reason for each verdict. "
              "Do not generate new graph edges.")
    reviews = []
    issues = []
    faithful = True
    chunks = [list(enumerate(relations))[i:i + 2] for i in range(0, len(relations), 2)]
    if not chunks:
        chunks = [[]]
    for part, chunk in enumerate(chunks, 1):
        part_path = folder / "source_audit_chunks" / f"part_{part:02d}.json"
        if part_path.exists():
            parsed = json.loads(part_path.read_text(encoding="utf-8"))
        else:
            user = ("<original>\n" + source + "\n</original>\n<plain>\n" + plain +
                    "\n</plain>\n<relations>\n" +
                    "\n".join(f"{i}: {s} | {p} | {o}" for i, (s, p, o) in chunk) +
                    "\n</relations>")
            errors = []
            for _ in range(2):
                try:
                    parsed, raw = completion(system, user, SOURCE_AUDIT_SCHEMA,
                                             "source_audit", max_tokens=2048)
                    expected_part = {i for i, _ in chunk}
                    reviewed_part = {r["index"] for r in parsed["edge_reviews"]}
                    if reviewed_part != expected_part:
                        raise ValueError(f"Audit part reviewed {sorted(reviewed_part)}; expected {sorted(expected_part)}")
                    parsed["raw_response"] = raw
                    save(part_path, parsed)
                    break
                except Exception as exc:
                    errors.append(repr(exc))
            else:
                raise RuntimeError(f"Source audit part {part} failed: {errors[-1]}")
        faithful = faithful and parsed["rephrase_faithful"]
        issues.extend(parsed["rephrase_issues"])
        reviews.extend(parsed["edge_reviews"])
    expected = set(range(len(relations)))
    seen = {r["index"] for r in reviews}
    if seen != expected:
        raise ValueError(f"Source audit reviewed {sorted(seen)}; expected {sorted(expected)}")
    save(path, {"rephrase_faithful": faithful, "rephrase_issues": sorted(set(issues)),
                "edge_reviews": sorted(reviews, key=lambda item: item["index"]),
                "relations": [list(r) for r in relations],
                "status": "machine_review_needs_human"})
    (folder / "source_audit_error.json").unlink(missing_ok=True)


def extract_plain_record(kg, record, folder):
    path = folder / "raw_graph.json"
    if path.exists():
        graph = Graph.from_file(str(path))
        plain = plain_rephrase(record["evidence_text"], folder)
        try:
            audit_against_original(record["evidence_text"], plain, graph, folder)
        except Exception as exc:
            save(folder / "source_audit_error.json", {"error": repr(exc)})
        return graph
    source = record["evidence_text"]
    save(folder / "input.json", {key: record[key] for key in
                                 ("record_id", "evidence_text", "sentence_ids", "provenance")})
    plain = plain_rephrase(source, folder)
    try:
        initial = kg.generate(input_data=plain, no_dspy=True, deduplication_method=None)
        (folder / "generation_error.json").unlink(missing_ok=True)
    except Exception as exc:
        save(folder / "generation_error.json", {"error": repr(exc),
                                                "next_step": "expanded_entities_relation_retry"})
        initial = Graph(entities=set(), relations=set(), edges=set())
    save(folder / "initial_graph.json", graph_data(initial))
    if initial.relations:
        graph = initial
        route = "kggen_plain"
    else:
        graph = repair_zero_edges(plain, initial, folder)
        route = "expanded_entities_relation_retry" if graph.relations else "zero_relations_after_retry"
    save(path, graph_data(graph))
    save(folder / "raw_counts.json", counts(graph))
    save(folder / "review_flags.json", {"record_id": record["record_id"],
                                        "route": route, "no_relations": not bool(graph.relations),
                                        "status": "candidate_requires_original_source_review"})
    try:
        audit_against_original(source, plain, graph, folder)
    except Exception as exc:
        save(folder / "source_audit_error.json", {"error": repr(exc)})
    return graph
