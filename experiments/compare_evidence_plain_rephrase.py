"""Compare original evidence sentences with fixed, source-checked plain rephrases."""

import json
from pathlib import Path

from kg_gen import KGGen
from kg_gen.models import Graph

from socnav_six_run_pilot import counts, graph_data, save


ROOT = Path("/data/code/Know2SocNav/data/kg-gen-socnav/runs")
SOURCE_RUN = ROOT / "pilot_5papers_20261020" / "papers"
OUTPUT = ROOT / "compare_original_plain_20261020"

# Fixed paraphrases preserve the source sentence's assertion and scope.
PLAIN = {
    "SOC-REV-002_ER_06639d9867af": "Robot movement is an important part of user acceptance. Studies evaluate how movement speed, observed distance, and cultural bias influence a user's perception of a robot.",
    "SOC-REV-002_ER_1e7ef8f73605": "Humans prefer to interact with robots that follow familiar rules of movement. Humans also prefer robots whose behavior they can predict intuitively.",
    "SOC-REV-006_ER_251deec4271a": "Social navigation enables robots to move safely and efficiently in environments shared with humans. In social navigation, robots respect social norms and prioritize human comfort.",
    "SOC-REV-006_ER_ead871ac7aad": "Social norms are informal rules that guide behavior in shared spaces. Social norms extend beyond collision avoidance and proxemics.",
    "SOC-REV-008_ER_4583636ab080": "SBCM assumes that a person's spatial responses are affected by the environment, other humans, or the person's location.",
    "SOC-REV-008_ER_f060d26d7fa4": "Proxemic rules are the four social zones. These social zones are elliptical regions around a person.",
    "SOC-REV-009_ER_ef364ef26ef2": "Later works propose that personal spaces are asymmetrical and dynamic.",
    "SOC-REV-009_ER_adb7b17105d6": "Hall divides personal space into concentric circular zones.",
    "SOC-REV-010_ER_26c6c84006b8": "Planning, navigation, and decision making have a more direct relationship with proxemics. These processes govern a robot's spatial position and movement trajectories relative to humans.",
    "SOC-REV-010_ER_b4e4d451a413": "Evaluation, safety, and communication are components of HRI mapping. These components do not directly determine interpersonal distance.",
}


def main():
    kg = KGGen(model="hosted_vllm/know2socnav-qwen3-30b-a3b", api_key="EMPTY",
               api_base="http://10.137.6.216:8000/v1", max_tokens=4096,
               temperature=0.0, retrieval_model="sentence-transformers/all-MiniLM-L6-v2")
    for index, (record_id, plain) in enumerate(PLAIN.items(), 1):
        paper_id = record_id.split("_ER_")[0]
        source = SOURCE_RUN / f"{paper_id}_recordwise" / "records" / record_id
        source_input = json.loads((source / "input.json").read_text(encoding="utf-8"))
        original = Graph.from_file(str(source / "raw_graph.json"))
        folder = OUTPUT / record_id
        save(folder / "inputs.json", {
            "record_id": record_id, "original": source_input["evidence_text"],
            "plain_rephrase": plain, "original_graph": str(source / "raw_graph.json"),
            "provenance": source_input["provenance"],
        })
        path = folder / "plain_graph.json"
        if path.exists():
            graph = Graph.from_file(str(path))
        else:
            try:
                graph = kg.generate(input_data=plain, no_dspy=True, deduplication_method=None)
            except Exception as exc:
                save(folder / "generation_error.json", {"error": repr(exc)})
                print(f"{index}/10 {record_id}: failed {exc!r}", flush=True)
                continue
            save(path, graph_data(graph))
        save(folder / "counts.json", {"original": counts(original), "plain_rephrase": counts(graph)})
        print(f"{index}/10 {record_id}: {len(original.relations)} -> {len(graph.relations)} relations", flush=True)


if __name__ == "__main__":
    main()
