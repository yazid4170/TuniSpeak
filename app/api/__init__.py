from fastapi import APIRouter

from app.api.feedback import router as feedback_router
from app.api.metrics import router as metrics_router
from app.api.routes import router as qa_router

__all__ = ["api_router"]

api_router_v1 = APIRouter(prefix="/api/v1")
api_router_v1.include_router(qa_router)
api_router_v1.include_router(feedback_router)
api_router_v1.include_router(metrics_router)

api_router = api_router_v1
