"""
main.py
-------
Application entrypoint. Wires together config, security, model, and database
modules into a FastAPI app with:
  - CORS whitelist
  - JWT + API-key auth
  - Rate limiting (slowapi)
  - Safe error handling (no stack traces leaked to clients)
  - JSON REST endpoints for the SPA dashboard
  - Jinja2 SSR route for lightweight hosting
  - WebSocket + SSE for real-time threat streaming

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import ipaddress
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import get_settings
from database import persistence
from model import inference_engine
from schemas import (
    BatchFlowIngest,
    FlowIngest,
    MLMetrics,
    SystemConfigUpdate,
    ThreatEvent,
    ThreatEventOut,
    ThreatSeverity,
    Token,
    TokenPayload,
    TokenRequest,
)
from security import (
    create_access_token,
    get_current_user,
    limiter,
    require_admin,
    verify_password,
    verify_sensor_api_key,
)

settings = get_settings()

logging.basicConfig(level=logging.INFO if not settings.DEBUG else logging.DEBUG)
logger = logging.getLogger("sentinel.main")

templates = Jinja2Templates(directory="app/templates")


# ---------------------------------------------------------------------------
# Demo user store — replace with a real database-backed user table in production.
# Password below is a bcrypt hash; never store plaintext passwords.
# ---------------------------------------------------------------------------
from security import hash_password  # noqa: E402

_DEMO_USERS = {
    "admin": {"password_hash": hash_password("change-this-password!"), "scope": "admin"},
}


# ---------------------------------------------------------------------------
# WebSocket connection manager for real-time threat broadcast
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self.active.append(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self.active:
                self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        dead: List[WebSocket] = []
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:  # noqa: BLE001 — client disconnected mid-send
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws)


manager = ConnectionManager()
# Simple in-process pub/sub queue that SSE clients each subscribe to independently.
_sse_subscribers: List[asyncio.Queue] = []


async def _publish_threat(event: ThreatEvent) -> None:
    payload = json.loads(event.model_dump_json())
    await manager.broadcast({"type": "threat", "data": payload})
    for q in list(_sse_subscribers):
        await q.put(payload)


# ---------------------------------------------------------------------------
# App lifespan — start/stop background workers cleanly
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    inference_engine.start()
    persistence.start_background_sync()
    logger.info("Sentinel backend started (device=%s, firebase_enabled=%s)",
                inference_engine.device, persistence.firestore.enabled)
    yield
    await inference_engine.stop()
    await persistence.stop_background_sync()
    logger.info("Sentinel backend shut down cleanly.")


app = FastAPI(
    title=settings.APP_NAME,
    docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,  # hide docs in prod
    redoc_url=None,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS: explicit origin whitelist, never "*" when credentials are allowed ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# ---------------------------------------------------------------------------
# Safe error handling — never leak stack traces / internals to clients
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.post("/auth/token", response_model=Token, tags=["auth"])
@limiter.limit("10/minute")  # brute-force protection on login
async def login(request: Request, credentials: TokenRequest):
    user = _DEMO_USERS.get(credentials.username)
    if not user or not verify_password(credentials.password, user["password_hash"]):
        # Deliberately generic message — don't reveal whether username exists.
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(subject=credentials.username, scope=user["scope"])
    return Token(access_token=token, expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)


# ---------------------------------------------------------------------------
# Ingestion (sensor -> backend), protected by API key, rate-limited higher
# since this is the high-throughput path from edge collectors.
# ---------------------------------------------------------------------------
def _classify_severity(score: float) -> ThreatSeverity:
    if score >= 0.6:
        return ThreatSeverity.CRITICAL
    if score >= 0.4:
        return ThreatSeverity.HIGH
    if score >= settings.ANOMALY_THRESHOLD:
        return ThreatSeverity.MEDIUM
    return ThreatSeverity.LOW


def _validate_ip(value: str) -> str:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid IP address: {value!r}")
    return value


async def _process_flow(flow: FlowIngest) -> ThreatEvent:
    _validate_ip(flow.src_ip)
    _validate_ip(flow.dst_ip)

    # Pad/truncate to the model's expected input dimension defensively.
    vec = flow.flow.features[: settings.MODEL_INPUT_DIM]
    vec = vec + [0.0] * (settings.MODEL_INPUT_DIM - len(vec))

    error, is_anomalous, confidence = await inference_engine.infer(vec)

    event = ThreatEvent(
        src_ip=flow.src_ip,
        dst_ip=flow.dst_ip,
        src_port=flow.src_port,
        dst_port=flow.dst_port,
        protocol=flow.protocol,
        sensor_id=flow.sensor_id,
        severity=_classify_severity(error) if is_anomalous else ThreatSeverity.LOW,
        score=round(error, 6),
        timestamp=flow.timestamp or datetime.now(timezone.utc),
        description="Autoencoder reconstruction-error anomaly" if is_anomalous else "Nominal flow",
    )

    if is_anomalous:
        await persistence.record_threat(event)
        await _publish_threat(event)

    return event


@app.post("/api/v1/flows", response_model=ThreatEventOut, tags=["ingestion"])
@limiter.limit(settings.RATE_LIMIT_INGEST)
async def ingest_flow(request: Request, flow: FlowIngest, _=Depends(verify_sensor_api_key)):
    """Ingest a single network flow, run real-time inference, persist+broadcast if anomalous."""
    event = await _process_flow(flow)
    return event


@app.post("/api/v1/flows/batch", response_model=List[ThreatEventOut], tags=["ingestion"])
@limiter.limit(settings.RATE_LIMIT_INGEST)
async def ingest_flow_batch(request: Request, batch: BatchFlowIngest, _=Depends(verify_sensor_api_key)):
    """Ingest multiple flows concurrently — the inference engine batches them internally."""
    results = await asyncio.gather(*(_process_flow(f) for f in batch.flows))
    return results


# ---------------------------------------------------------------------------
# Read endpoints for the dashboard (JWT-protected, JSON)
# ---------------------------------------------------------------------------
@app.get("/api/v1/threats", response_model=List[ThreatEventOut], tags=["threats"])
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_threats(request: Request, limit: int = 50, user: TokenPayload = Depends(get_current_user)):
    limit = max(1, min(limit, 500))  # clamp to prevent abuse
    return persistence.get_recent_threats(limit=limit)


@app.get("/api/v1/metrics", response_model=MLMetrics, tags=["metrics"])
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def get_metrics(request: Request, user: TokenPayload = Depends(get_current_user)):
    return MLMetrics(
        total_flows_processed=inference_engine.total_processed,
        total_threats_detected=persistence.total_threats_detected,
        avg_inference_latency_ms=inference_engine.avg_latency_ms,
        model_device=str(inference_engine.device),
        queue_depth=inference_engine.queue_depth,
        uptime_seconds=round(inference_engine.uptime_seconds, 1),
    )


@app.patch("/api/v1/system/config", tags=["system"])
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def update_system_config(request: Request, update: SystemConfigUpdate, user: TokenPayload = Depends(require_admin)):
    """Admin-only: adjust runtime-tunable parameters (e.g. anomaly threshold)."""
    if update.anomaly_threshold is not None:
        settings.ANOMALY_THRESHOLD = update.anomaly_threshold
    return {"status": "updated", "anomaly_threshold": settings.ANOMALY_THRESHOLD}


# ---------------------------------------------------------------------------
# Real-time streaming: WebSocket + Server-Sent Events
# ---------------------------------------------------------------------------
@app.websocket("/ws/threats")
async def ws_threats(websocket: WebSocket):
    """
    Real-time threat stream for the dashboard. Token is passed as a query param
    since browsers can't set custom headers on the WebSocket handshake; validate
    it before accepting to avoid handing a socket to unauthenticated clients.
    """
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=4401)
        return
    try:
        from app.security import decode_access_token
        decode_access_token(token)
    except HTTPException:
        await websocket.close(code=4401)
        return

    await manager.connect(websocket)
    try:
        while True:
            # We only push server->client; discard any client pings/keepalives.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@app.get("/sse/threats", tags=["streaming"])
async def sse_threats(request: Request, user: TokenPayload = Depends(get_current_user)):
    """Server-Sent Events alternative to WebSocket for simpler HTTP/1.1-only clients."""
    queue: asyncio.Queue = asyncio.Queue()
    _sse_subscribers.append(queue)

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"event: threat\ndata: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"  # comment line prevents proxy timeouts
        finally:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Server-side rendered dashboard (Jinja2) — for lightweight/no-JS hosting
# ---------------------------------------------------------------------------
@app.get("/dashboard", tags=["ssr"])
@limiter.limit(settings.RATE_LIMIT_DEFAULT)
async def dashboard(request: Request):
    recent = persistence.get_recent_threats(limit=25)
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"threats": recent, "app_name": settings.APP_NAME},
    )


@app.get("/healthz", tags=["system"])
async def healthz():
    """Unauthenticated liveness probe for orchestrators/load balancers."""
    return {"status": "ok"}
