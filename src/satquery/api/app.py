"""FastAPI application: ``POST /analyze``, ``GET /tools``, ``GET /health``, result files, UI.

Run with ``just serve`` (serves the built UI at ``/`` when ``ui/dist`` exists).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from satquery import __version__
from satquery.agent.pipeline import analyze
from satquery.api.schemas import AnalyzeResponse, ArtifactOut, ErrorResponse, HealthResponse
from satquery.api.storage import RequestStore
from satquery.config import Settings, get_settings
from satquery.errors import ErrorCode, ErrorInfo, SatQueryError
from satquery.io.dates import parse_iso_date
from satquery.io.validate import InputOverrides
from satquery.tools.registry import default_registry
from satquery.types import BandRole, Modality

MAX_FILES = 2
HTTP_STATUS = {"ok": 200, "partial": 200, "rejected": 422, "error": 500}


def _error_response(info: ErrorInfo, status: int = 422) -> JSONResponse:
    return JSONResponse(ErrorResponse(error=info).model_dump(mode="json"), status_code=status)


def _parse_overrides(
    modality: str | None, acquired: str | None, band_map: str | None, index: int
) -> InputOverrides:
    field = f"file {index}"
    try:
        parsed_date: date | None = parse_iso_date(acquired) if acquired else None
    except ValueError as exc:
        raise SatQueryError(
            ErrorCode.INVALID_OVERRIDE,
            f"{field}: date must be YYYY-MM-DD, got {acquired!r}.",
            details={"file_index": index, "date": acquired},
        ) from exc
    bands: dict[BandRole, int] | None = None
    if band_map:
        try:
            raw = json.loads(band_map)
            if not isinstance(raw, dict):
                raise ValueError("not an object")
            bands = {BandRole(str(k).lower()): int(v) for k, v in raw.items()}
        except (ValueError, TypeError) as exc:
            raise SatQueryError(
                ErrorCode.INVALID_OVERRIDE,
                f'{field}: band_map must be a JSON object like {{"green": 2, "nir": 4}} '
                f"with roles {[r.value for r in BandRole]}.",
                details={"file_index": index, "band_map": band_map, "reason": str(exc)},
            ) from exc
    try:
        mod = Modality(modality.lower()) if modality else None
    except ValueError as exc:
        raise SatQueryError(
            ErrorCode.INVALID_OVERRIDE,
            f"{field}: modality must be one of optical, multispectral, sar.",
            details={"file_index": index, "modality": modality},
        ) from exc
    return InputOverrides(modality=mod, acquired=parsed_date, band_map=bands)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    registry = default_registry()
    store = RequestStore(settings.data_dir, settings.max_upload_mb * (1 << 20))
    app = FastAPI(
        title="SatQuery AI",
        version=__version__,
        description="Agentic vision-language assistant for satellite imagery.",
    )

    def file_url(request_id: str, name: str) -> str:
        return f"/requests/{request_id}/files/{name}"

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        available, reason = registry.get("qwen_vl_zeroshot").is_available()
        return HealthResponse(
            status="ok",
            version=__version__,
            tools=len(registry),
            vlm={"model_id": settings.vlm_model_id, "available": available, "reason": reason},
        )

    @app.get("/tools")
    def tools() -> list[dict[str, Any]]:
        return registry.export_json()

    @app.post(
        "/analyze",
        response_model=AnalyzeResponse,
        responses={413: {"model": ErrorResponse}, 422: {"model": AnalyzeResponse}},
    )
    def analyze_endpoint(
        files: Annotated[list[UploadFile], File(description="1 or 2 GeoTIFFs")],
        query: Annotated[str, Form(min_length=1, max_length=2000)],
        modality_1: Annotated[str | None, Form()] = None,
        modality_2: Annotated[str | None, Form()] = None,
        date_1: Annotated[str | None, Form(description="YYYY-MM-DD")] = None,
        date_2: Annotated[str | None, Form(description="YYYY-MM-DD")] = None,
        band_map_1: Annotated[str | None, Form(description='JSON, e.g. {"nir": 4}')] = None,
        band_map_2: Annotated[str | None, Form()] = None,
        benchmark: Annotated[bool, Form()] = False,
    ) -> JSONResponse:
        if not 1 <= len(files) <= MAX_FILES:
            code = ErrorCode.NO_INPUTS if not files else ErrorCode.TOO_MANY_INPUTS
            msg = f"Upload 1 or {MAX_FILES} images, got {len(files)}."
            return _error_response(ErrorInfo(code=code, message=msg))
        try:
            raw = [(modality_1, date_1, band_map_1), (modality_2, date_2, band_map_2)]
            overrides = [_parse_overrides(*raw[i], index=i + 1) for i in range(len(files))]
            request_id, request_dir = store.new_request()
            paths = [
                store.save_upload(request_dir, i + 1, f.filename or f"image{i + 1}.tif", f.file)
                for i, f in enumerate(files)
            ]
        except SatQueryError as exc:
            status = 413 if exc.code is ErrorCode.UPLOAD_TOO_LARGE else 422
            return _error_response(exc.info, status)

        result = analyze(
            paths,
            query,
            overrides,  # type: ignore[arg-type]
            benchmark=benchmark,
            out_dir=request_dir / "outputs",
            registry=registry,
            request_id=request_id,
        )
        trace = result.trace
        body = AnalyzeResponse(
            request_id=request_id,
            status=trace.status,
            answer=trace.answer,
            confidence=trace.answer.confidence if trace.answer else None,
            error=trace.errors[0] if trace.status in ("rejected", "error") else None,
            artifacts=[
                ArtifactOut(
                    **a.model_dump(exclude={"filename"}), url=file_url(request_id, a.filename)
                )
                for a in trace.artifacts
            ],
            trace_url=file_url(request_id, "trace.json"),
            trace=trace,
        )
        return JSONResponse(body.model_dump(mode="json"), status_code=HTTP_STATUS[trace.status])

    @app.get("/requests/{request_id}/files/{name}")
    def request_file(request_id: str, name: str) -> FileResponse:
        path = store.output_file(request_id, name)
        if path is None:
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(path)

    ui_dir = Path(settings.ui_dir)
    if (ui_dir / "index.html").is_file():
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
    return app


app = create_app()
