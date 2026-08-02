"""
schemas.py
----------
All request/response models. Pydantic enforces type validation, bounds checking,
and sanitization at the boundary — never trust raw client input past this layer.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ThreatSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Protocol(str, Enum):
    TCP = "TCP"
    UDP = "UDP"
    ICMP = "ICMP"
    OTHER = "OTHER"


# ---------------------------------------------------------------------------
# Ingestion (sensor -> backend)
# ---------------------------------------------------------------------------

class FlowFeatures(BaseModel):
    """
    A single fixed-length numeric feature vector describing one network flow
    (e.g. duration, byte counts, packet counts, flags — however the upstream
    feature extractor is built). Bounds are enforced to reject garbage/oversized
    payloads before they ever reach the model.
    """
    features: List[float] = Field(..., min_length=1, max_length=256)

    @field_validator("features")
    @classmethod
    def finite_values_only(cls, v: List[float]) -> List[float]:
        for x in v:
            if x != x or x in (float("inf"), float("-inf")):  # NaN / Inf check
                raise ValueError("feature vector contains NaN or Inf")
        return v


class FlowIngest(BaseModel):
    """Payload submitted by an edge sensor/collector for a single observed flow."""
    src_ip: str = Field(..., max_length=45)  # IPv4/IPv6 max length
    dst_ip: str = Field(..., max_length=45)
    src_port: int = Field(..., ge=0, le=65535)
    dst_port: int = Field(..., ge=0, le=65535)
    protocol: Protocol = Protocol.TCP
    sensor_id: str = Field(..., max_length=64)
    timestamp: Optional[datetime] = None
    flow: FlowFeatures

    @field_validator("src_ip", "dst_ip")
    @classmethod
    def basic_ip_sanity(cls, v: str) -> str:
        # Lightweight sanity check; real IP parsing done via ipaddress in the route handler.
        if any(c.isspace() for c in v) or len(v) == 0:
            raise ValueError("invalid IP address format")
        return v


class BatchFlowIngest(BaseModel):
    flows: List[FlowIngest] = Field(..., min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Inference / Threat output
# ---------------------------------------------------------------------------

class InferenceResult(BaseModel):
    reconstruction_error: float
    is_anomalous: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


class ThreatEvent(BaseModel):
    id: Optional[str] = None
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: Protocol
    sensor_id: str
    severity: ThreatSeverity
    score: float
    timestamp: datetime
    description: str = Field(default="", max_length=500)


class ThreatEventOut(ThreatEvent):
    """Response model — safe to serialize back to clients."""
    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class MLMetrics(BaseModel):
    total_flows_processed: int
    total_threats_detected: int
    avg_inference_latency_ms: float
    model_device: str
    queue_depth: int
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class TokenRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8, max_length=128)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    sub: str
    exp: int
    scope: str = "user"


# ---------------------------------------------------------------------------
# System control
# ---------------------------------------------------------------------------

class SystemConfigUpdate(BaseModel):
    anomaly_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    firebase_sync_enabled: Optional[bool] = None
