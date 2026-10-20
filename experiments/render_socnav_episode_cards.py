"""Render KGGen episode graphs beside the exact source text for review."""

import argparse
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = PROJECT_ROOT / "data/episode_resources/runs/call01_freeze_final"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
ROLES = (
    ("observation", "OBSERVATION", "#e8f3ff"),
    ("prediction", "PREDICTION", "#fff5df"),
    ("reasoning", "REASONING", "#f0ebff"),
    ("action", "ACTION", "#e6f7ed"),
)


def font(size, bold=False):
    return ImageFont.truetype(BOLD_PATH if bold else FONT_PATH, size)


def wrap(draw, text, text_font, width):
    lines = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=text_font) <= width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        lines.append(line)
    return lines


def source_sections(episode_id):
    path = SOURCE_ROOT / f"{episode_id}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    propositions = data["call_0_normalization"]["propositions"]
    result = {}
    for proposition in propositions:
        role = proposition["role"]
        key = (proposition["source_part_id"], proposition["source_field"])
        result.setdefault(role, {})[key] = proposition["source_text"]
    return {role: "\n".join(parts.values()) for role, parts in result.items()}


def dot_quote(value):
    return json.dumps(str(value), ensure_ascii=False)


def short(value, limit=56):
    value = " ".join(str(value).split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def graph_image(graph, graph_width, graph_height):
    relations = [tuple(x) for x in graph["relations"]]
    if not relations:
        return None
    endpoints = sorted({item for subject, _, obj in relations for item in (subject, obj)})
    node_ids = {name: f"n{i}" for i, name in enumerate(endpoints)}
    lines = [
        "digraph Episode {",
        'graph [rankdir=LR, bgcolor="transparent", nodesep=0.35, ranksep=1.1, '
        'splines=true, overlap=false, outputorder=edgesfirst, pad=0.25];',
        'node [shape=box, style="rounded,filled", fillcolor="#eaf3ff", '
        'color="#4d82ad", fontname="DejaVu Sans", fontsize=16, margin="0.18,0.12"];',
        'edge [color="#6486a4", fontcolor="#38536a", fontname="DejaVu Sans", '
        'fontsize=12, arrowsize=0.6];',
    ]
    for name in endpoints:
        lines.append(f'{node_ids[name]} [label={dot_quote(short(name, 42))}];')
    for subject, predicate, obj in sorted(relations):
        lines.append(f'{node_ids[subject]} -> {node_ids[obj]} '
                     f'[label={dot_quote(short(predicate, 36))}];')
    lines.append("}")
    result = subprocess.run(["dot", "-Tpng", "-Gdpi=170"],
                            input="\n".join(lines).encode("utf-8"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=True)
    from io import BytesIO
    picture = Image.open(BytesIO(result.stdout)).convert("RGBA")
    picture.thumbnail((graph_width, graph_height), Image.Resampling.LANCZOS)
    return picture


def render_episode(episode_dir, output_dir):
    episode_id = episode_dir.name
    clustered = episode_dir / "clustered_graph.json"
    raw = episode_dir / "raw_graph.json"
    stage = ("clustered" if clustered.exists() else
             "raw / cluster failed" if (episode_dir / "cluster_error.json").exists() else
             "raw / not clustered" if raw.exists() else "generation failed")
    graph = (json.loads((clustered if clustered.exists() else raw).read_text(encoding="utf-8"))
             if raw.exists() else {"entities": [], "relations": []})
    sections = source_sections(episode_id)
    relations = graph["relations"]
    entities = set(graph["entities"])
    endpoints = {item for subject, _, obj in relations for item in (subject, obj)}
    isolated = sorted(entities - endpoints)

    width, height = 3000, 1900
    image = Image.new("RGB", (width, height), "#f5f7fa")
    draw = ImageDraw.Draw(image)
    title_font = font(45, True)
    body_font = font(25)
    small_font = font(21)
    label_font = font(23, True)
    draw.text((64, 43), episode_id, fill="#152433", font=title_font)
    draw.text((64, 105), f"KGGen candidate | {stage} | {len(entities)} entities | "
              f"{len(relations)} relations | {len(isolated)} isolated", fill="#4b5d6c", font=small_font)

    left_x, left_y, left_w = 64, 170, 1130
    draw.rounded_rectangle((left_x, left_y, left_x + left_w, height - 90),
                           radius=24, fill="white", outline="#d9e1e8", width=2)
    draw.text((left_x + 32, left_y + 27), "ORIGINAL EPISODE TEXT", fill="#17314b", font=label_font)
    y = left_y + 80
    for role, label, color in ROLES:
        text = sections.get(role, "")
        if not text:
            continue
        lines = wrap(draw, text, body_font, left_w - 78)
        box_h = 46 + 34 * len(lines) + 23
        draw.rounded_rectangle((left_x + 23, y, left_x + left_w - 23, y + box_h),
                               radius=15, fill=color)
        draw.text((left_x + 42, y + 11), label, fill="#24435e", font=label_font)
        for i, line in enumerate(lines):
            draw.text((left_x + 42, y + 48 + i * 34), line, fill="#162c3f", font=body_font)
        y += box_h + 18

    right_x, right_y, right_w = 1228, 170, 1708
    draw.rounded_rectangle((right_x, right_y, right_x + right_w, height - 90),
                           radius=24, fill="white", outline="#d9e1e8", width=2)
    draw.text((right_x + 32, right_y + 27), "SITUATED GRAPH CANDIDATE", fill="#17314b", font=label_font)
    graph_box = (right_x + 35, right_y + 85, right_w - 70, 1380)
    picture = graph_image(graph, graph_box[2], graph_box[3])
    if picture:
        px = graph_box[0] + (graph_box[2] - picture.width) // 2
        py = graph_box[1] + (graph_box[3] - picture.height) // 2
        image.paste(picture, (px, py), picture)
    else:
        draw.text((right_x + 70, right_y + 170),
                  "NO RELATIONS EXTRACTED" if raw.exists() else "GENERATION FAILED",
                  fill="#a34032", font=font(38, True))
        draw.text((right_x + 70, right_y + 240),
                  "KGGen returned entities, but no source-supported triples." if raw.exists()
                  else "KGGen did not return a valid graph for this episode.",
                  fill="#425b70", font=body_font)
        chips = ", ".join(short(name, 32) for name in sorted(entities))
        for i, line in enumerate(wrap(draw, chips, body_font, right_w - 150)):
            draw.text((right_x + 70, right_y + 325 + i * 38), line,
                      fill="#425b70", font=body_font)
    if isolated and relations:
        draw.text((right_x + 48, height - 175),
                  "Unlinked entities: " + ", ".join(short(x, 24) for x in isolated[:7]),
                  fill="#5c6975", font=small_font)
    draw.text((64, height - 56),
              "Exact source fields from episode_resources. KGGen graph is an unreviewed candidate; no relation was added for display.",
              fill="#526577", font=small_font)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{episode_id}.png"
    image.save(path, optimize=True)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    output = args.run / "episode_cards"
    for episode_dir in sorted((args.run / "episodes").iterdir()):
        if episode_dir.is_dir():
            path = render_episode(episode_dir, output)
            if path:
                print(path, flush=True)


if __name__ == "__main__":
    main()
