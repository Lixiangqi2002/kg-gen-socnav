from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json
import os
import re

import logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.kg_gen.kg_gen import KGGen
from src.kg_gen.models import Graph
from kg_gen.steps._3_deduplicate import DeduplicateMethod
from src.kg_gen.utils.visualize_kg import _build_view_model

APP_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = (
    APP_DIR.parent / "src" / "kg_gen" / "utils" / "template.html"
).resolve()
DATA_ROOT = (APP_DIR.parent / "app" / "examples").resolve()


@dataclass(frozen=True)
class ExampleGraph:
    slug: str
    title: str
    path: Path
    wiki_url: str


# TODO: this will be read from huggingface once it is uploaded.
EXAMPLE_GRAPHS: tuple[ExampleGraph, ...] = ()
for file in DATA_ROOT.glob("*.json"):
    EXAMPLE_GRAPHS += (
        ExampleGraph(
            slug=file.stem,
            title=file.stem,
            path=file,
            wiki_url=f"https://en.wikipedia.org/wiki/{file.stem}",
        ),
    )


EXAMPLE_INDEX = {
    example.slug: example for example in EXAMPLE_GRAPHS if example.path.exists()
}

if len(EXAMPLE_INDEX) < len(EXAMPLE_GRAPHS):
    missing = [
        example.slug for example in EXAMPLE_GRAPHS if example.slug not in EXAMPLE_INDEX
    ]
    logger = logging.getLogger("kg_gen_app")
    logger.warning("Example graphs missing on disk: %s", ", ".join(missing))

if not TEMPLATE_PATH.exists():
    raise RuntimeError(f"Template not found at {TEMPLATE_PATH}")

logger = logging.getLogger("kg_gen_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


app = FastAPI(title="kg-gen explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
REMOTE_QWEN_MODEL = "hosted_vllm/know2socnav-qwen3-30b-a3b"
REMOTE_QWEN_API_BASE = os.environ.get(
    "KGGEN_REMOTE_QWEN_API_BASE", "http://10.137.6.216:8000/v1"
)
SOCNAV_RUNS_ROOT = Path(os.environ.get(
    "KGGEN_SOCNAV_RUNS_ROOT", "/data/code/Know2SocNav/data/kg-gen-socnav/runs"
)).resolve()


def _socnav_run_dir(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise HTTPException(status_code=400, detail="Invalid run ID")
    run_dir = SOCNAV_RUNS_ROOT / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run not found")
    return run_dir


def _socnav_graph_path(run_dir: Path, graph_id: str, stage: str) -> Path:
    if stage not in {"raw", "clustered"}:
        raise HTTPException(status_code=400, detail="Invalid graph stage")
    if graph_id == "evidence_100":
        folder = run_dir / "evidence_100"
    elif re.fullmatch(r"(?:SNEI|MUSON)_EPISODE_[a-zA-Z0-9]+", graph_id):
        folder = run_dir / "episodes" / graph_id
    else:
        raise HTTPException(status_code=400, detail="Invalid graph ID")
    path = folder / f"{stage}_graph.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Graph is not ready")
    return path


@app.get("/api/socnav/runs")
async def list_socnav_runs() -> JSONResponse:
    runs = []
    if SOCNAV_RUNS_ROOT.is_dir():
        for run_dir in sorted(SOCNAV_RUNS_ROOT.iterdir(), reverse=True):
            if not run_dir.is_dir() or not re.fullmatch(r"[A-Za-z0-9_-]+", run_dir.name):
                continue
            folders = [run_dir / "evidence_100"]
            episodes = run_dir / "episodes"
            if episodes.is_dir():
                folders.extend(sorted(p for p in episodes.iterdir() if p.is_dir()))
            graphs = []
            for folder in folders:
                if not folder.is_dir():
                    continue
                stage = next((s for s in ("clustered", "raw")
                              if (folder / f"{s}_graph.json").is_file()), None)
                path = folder / f"{stage}_graph.json" if stage else None
                status = ("cluster_failed" if stage == "raw" and
                          (folder / "cluster_error.json").is_file() else stage)
                if status is None:
                    status = ("generation_failed" if (folder / "generation_error.json").is_file()
                              else "pending")
                graphs.append({"id": folder.name, "status": status,
                               "stage": stage, "modified": path.stat().st_mtime_ns if path else 0})
            if graphs:
                runs.append({"id": run_dir.name, "graphs": graphs})
    return JSONResponse(runs)


@app.get("/api/socnav/runs/{run_id}/graphs/{graph_id}")
async def load_socnav_graph(run_id: str, graph_id: str, stage: str = "clustered") -> JSONResponse:
    path = _socnav_graph_path(_socnav_run_dir(run_id), graph_id, stage)
    try:
        graph = Graph.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise HTTPException(status_code=500, detail=f"Saved graph is invalid: {exc}") from exc
    return JSONResponse({"graph": graph.model_dump(mode="json"),
                         "view": _build_view_model(graph), "stage": stage})


@app.get("/api/socnav/runs/{run_id}/graphs/{graph_id}/visualization", response_class=HTMLResponse)
async def visualize_socnav_graph(run_id: str, graph_id: str, stage: str = "clustered") -> HTMLResponse:
    path = _socnav_graph_path(_socnav_run_dir(run_id), graph_id, stage)
    try:
        graph = Graph.model_validate_json(path.read_text(encoding="utf-8"))
        view = _build_view_model(graph)
    except (OSError, ValidationError) as exc:
        raise HTTPException(status_code=500, detail=f"Saved graph is invalid: {exc}") from exc
    payload = json.dumps(view, ensure_ascii=False).replace("<", "\\u003c")
    html = TEMPLATE_PATH.read_text(encoding="utf-8").replace("<!--DATA-->", payload)
    return HTMLResponse(html)


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index_path = APP_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html missing")
    logger.debug("Serving index page")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/template")
async def serve_template() -> FileResponse:
    logger.debug("Serving visualization template from %s", TEMPLATE_PATH)
    return FileResponse(TEMPLATE_PATH, media_type="text/html")


@app.get("/api/examples")
async def list_examples() -> JSONResponse:
    logger.debug("Listing built-in example graphs")
    items = [
        {"slug": example.slug, "title": example.title, "wiki_url": example.wiki_url}
        for example in sorted(
            EXAMPLE_INDEX.values(), key=lambda item: item.title.lower()
        )
    ]
    return JSONResponse(items)


@app.get("/api/examples/{slug}")
async def load_example(slug: str) -> JSONResponse:
    example = EXAMPLE_INDEX.get(slug)
    if example is None:
        raise HTTPException(status_code=404, detail=f"Example '{slug}' not found")

    if not example.path.exists():
        logger.error("Example graph missing: %s", example.path)
        raise HTTPException(status_code=404, detail=f"Example '{slug}' is unavailable")

    try:
        payload = json.loads(example.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse example graph %s", slug)
        raise HTTPException(
            status_code=500, detail=f"Example '{slug}' is invalid: {exc}"
        )

    logger.info("Loaded example graph '%s' from %s", slug, example.path)
    return JSONResponse(payload)


@app.post("/api/graph/view")
async def build_view(graph: Graph) -> JSONResponse:
    """Convert a raw KGGen graph payload into the template view model."""
    logger.info(
        "Received request to build view: entities=%s relations=%s",
        len(graph.entities),
        len(graph.relations),
    )
    try:
        view = _build_view_model(graph)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to build view model")
        raise HTTPException(status_code=500, detail=f"Failed to build view: {exc}")
    logger.info(
        "View model ready: nodes=%s edges=%s", len(view["nodes"]), len(view["edges"])
    )
    return JSONResponse({"view": view, "graph": graph.model_dump(mode="json")})


def _clean_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"true", "1", "yes", "on"}


@app.post("/api/generate")
async def generate_graph(
    api_key: str = Form("", description="Optional API key for cloud models"),
    model: str = Form("ollama_chat/qwen3:4b"),
    context: Optional[str] = Form(None),
    chunk_size: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    cluster: Optional[str] = Form(None),
    source_text: Optional[str] = Form(None),
    text_file: Optional[UploadFile] = File(None),
    retrieval_model: Optional[str] = Form("sentence-transformers/all-MiniLM-L6-v2"),
) -> JSONResponse:
    text_fragments: list[str] = []

    cleaned_text = _clean_str(source_text)
    if cleaned_text:
        text_fragments.append(cleaned_text)

    if text_file is not None:
        try:
            contents = await text_file.read()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to read uploaded text file")
            raise HTTPException(
                status_code=400, detail=f"Reading text file failed: {exc}"
            )
        try:
            decoded = contents.decode("utf-8")
        except UnicodeDecodeError as exc:
            logger.exception("Uploaded text file must be UTF-8")
            raise HTTPException(
                status_code=400, detail=f"Text file must be UTF-8: {exc}"
            )
        cleaned_file_text = _clean_str(decoded)
        if cleaned_file_text:
            text_fragments.append(cleaned_file_text)

    if not text_fragments:
        raise HTTPException(
            status_code=400, detail="Provide inline text or upload a .txt file"
        )

    request_text = "\n\n".join(text_fragments)

    numeric_chunk: Optional[int] = None
    if chunk_size:
        try:
            numeric_chunk = int(chunk_size)
        except ValueError as exc:
            logger.warning("Invalid chunk_size received: %s", chunk_size)
            raise HTTPException(
                status_code=400, detail=f"chunk_size must be an integer: {exc}"
            )
        if numeric_chunk <= 0:
            numeric_chunk = None

    numeric_temperature: Optional[float] = None
    if temperature:
        try:
            numeric_temperature = float(temperature)
        except ValueError as exc:
            logger.warning("Invalid temperature received: %s", temperature)
            raise HTTPException(
                status_code=400, detail=f"temperature must be numeric: {exc}"
            )

    # Validate temperature for gpt-5 family models
    if "gpt-5" in model:
        if numeric_temperature is not None and numeric_temperature < 1.0:
            raise HTTPException(
                status_code=400,
                detail="Temperature must be 1.0 or higher for gpt-5 family models",
            )
        # Set default temperature to 1.0 for gpt-5 models if not specified
        if numeric_temperature is None:
            numeric_temperature = 1.0

    remote_qwen = model == REMOTE_QWEN_MODEL
    kg_gen = KGGen(
        model=model,
        max_tokens=4096 if remote_qwen else 16000,
        api_key="EMPTY" if remote_qwen else api_key,
        api_base=REMOTE_QWEN_API_BASE if remote_qwen else None,
        temperature=numeric_temperature if numeric_temperature is not None else 0.0,
        retrieval_model=retrieval_model,
    )

    logger.info(
        "Generating graph via KGGen: model=%s cluster=%s chunk_size=%s context_len=%s text_len=%s temperature=%s retrieval_model=%s",
        model,
        _parse_bool(cluster),
        numeric_chunk,
        len((_clean_str(context) or "")),
        len(request_text),
        numeric_temperature,
        retrieval_model,
    )
    try:
        graph = kg_gen.generate(
            input_data=request_text,
            context=_clean_str(context) or "",
            chunk_size=numeric_chunk,
            temperature=numeric_temperature,
            no_dspy=remote_qwen,
            deduplication_method=None if remote_qwen else DeduplicateMethod.SEMHASH,
        )
        if _parse_bool(cluster):
            graph = kg_gen.cluster(graph, context=_clean_str(context) or "")
    except ValidationError as exc:
        logger.exception("KGGen returned validation error")
        raise HTTPException(status_code=400, detail=f"Invalid graph result: {exc}")
    except Exception as exc:
        logger.exception("KGGen generation failed")
        raise HTTPException(status_code=500, detail=f"KGGen failed: {exc}")

    try:
        view = _build_view_model(graph)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to build view model after generation")
        raise HTTPException(status_code=500, detail=f"Failed to build view: {exc}")

    logger.info(
        "Graph generation complete: entities=%s relations=%s",
        len(graph.entities),
        len(graph.relations),
    )
    return JSONResponse({"view": view, "graph": graph.model_dump(mode="json")})


# Serve static files (CSS, JS, etc.) - must be mounted after all routes
app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="static")
