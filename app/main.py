from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.api import api_router
from app.core.config import get_settings

logger = structlog.get_logger(__name__)


def configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("lifespan.start", environment=settings.environment)
    try:
        yield
    finally:
        logger.info("lifespan.stop")


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()

    application = FastAPI(
        title="TuniSpeak API",
        version="0.1.0",
        description="Trilingual QA assistant backend",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router)

    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.exists():

        @application.get("/", include_in_schema=False)
        async def serve_index() -> FileResponse:
            return FileResponse(frontend_dir / "index.html")

        @application.get("/app.js", include_in_schema=False)
        async def serve_app_js() -> FileResponse:
            return FileResponse(frontend_dir / "app.js", media_type="application/javascript")

        @application.get("/styles.css", include_in_schema=False)
        async def serve_styles() -> FileResponse:
            return FileResponse(frontend_dir / "styles.css", media_type="text/css")

    return application


app = create_app()


def run() -> None:
    """Entry point for poetry script."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
