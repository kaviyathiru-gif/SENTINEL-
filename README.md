# Sentinel ML-IDS Backend

A modular FastAPI + PyTorch backend for a network-flow anomaly-detection (IDS)
dashboard. Designed to run identically on a server or resource-constrained
edge hardware (Raspberry Pi, Jetson Nano, gateway appliances).

## Project layout

```
sentinel-backend/
├── app/
│   ├── main.py        # FastAPI app, routes, middleware, WebSocket/SSE
│   ├── model.py        # PyTorch autoencoder + async batching inference engine
│   ├── database.py     # Firestore sync + SQLite/ring-buffer offline fallback
│   ├── security.py     # JWT auth, API-key auth, rate limiting
│   ├── config.py        # pydantic-settings environment configuration
│   ├── schemas.py        # Pydantic request/response models
│   └── templates/
│       └── dashboard.html   # Jinja2 SSR view
├── scripts/
│   └── init_model_weights.py   # generates a starter checkpoint
├── requirements.txt
├── .env.example
└── README.md
```

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
```

Edit `.env` and set **strong, unique** values for `JWT_SECRET_KEY` and
`SENSOR_API_KEY` (never commit real values):

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

For edge/CPU-only deployment (Raspberry Pi, Jetson Nano), install the CPU
build of PyTorch instead of the default:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Set `MODEL_DEVICE=cpu` (or leave `auto`, which detects CUDA automatically).

## 2. Initialize a model checkpoint

The API will run with a randomly-initialized model if no checkpoint exists
(useful for wiring/testing), but you should generate a starter checkpoint and
later replace it with one trained on real benign-flow data:

```bash
python -m scripts.init_model_weights
```

To retrain: build a training script that instantiates `app.model.NeuralNetwork`,
trains on normalized benign flow-feature vectors with MSE reconstruction loss,
and `torch.save(model.state_dict(), settings.MODEL_PATH)`.

## 3. (Optional) Firebase setup

1. Create a Firebase project and a service account with Firestore access.
2. Download the service account JSON to `secrets/firebase-service-account.json`
   (already gitignored — never commit this file).
3. In `.env`, set `FIREBASE_ENABLED=true` and `FIREBASE_PROJECT_ID=<your-project-id>`.

If Firebase is disabled or unreachable, all data still persists locally via
SQLite (or an in-memory ring buffer if disk is unavailable) and syncs
automatically once connectivity is restored.

## 4. Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API docs (disabled automatically when `ENVIRONMENT=production`):
`http://localhost:8000/api/docs`

## 5. API overview

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/auth/token` | none (rate-limited) | Exchange username/password for a JWT |
| POST | `/api/v1/flows` | `X-API-Key` | Ingest one flow, run inference |
| POST | `/api/v1/flows/batch` | `X-API-Key` | Ingest up to 500 flows at once |
| GET | `/api/v1/threats` | Bearer JWT | Recent threat events (JSON, for SPA) |
| GET | `/api/v1/metrics` | Bearer JWT | Live ML/system metrics |
| PATCH | `/api/v1/system/config` | Bearer JWT (admin scope) | Tune runtime config |
| GET | `/dashboard` | none* | Server-rendered (Jinja2) threat table |
| WS | `/ws/threats?token=<jwt>` | JWT via query param | Real-time threat push |
| GET | `/sse/threats` | Bearer JWT | Server-Sent Events threat stream |
| GET | `/healthz` | none | Liveness probe |

\* Add `get_current_user`/`require_admin` to `/dashboard` if it should be
private in your deployment — it's left open here as a lightweight public
status page example; adjust to your threat model.

### Example: ingest a flow

```bash
curl -X POST http://localhost:8000/api/v1/flows \
  -H "X-API-Key: $SENSOR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "src_ip": "10.0.0.5",
    "dst_ip": "10.0.0.1",
    "src_port": 443,
    "dst_port": 51322,
    "protocol": "TCP",
    "sensor_id": "edge-gw-01",
    "flow": {"features": [0.1, 0.2, 0.0, 0.5, ...]}
  }'
```

### Example: get a JWT and fetch threats

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"change-this-password!"}' | jq -r .access_token)

curl http://localhost:8000/api/v1/threats -H "Authorization: Bearer $TOKEN"
```

**Change the demo admin password in `app/main.py` (`_DEMO_USERS`) before any
real deployment** — it exists only to demonstrate the auth flow. Replace it
with a real user table backed by your database.

## 6. Security notes

- CORS origins are an explicit whitelist (`CORS_ORIGINS` in `.env`) — never `*`
  in combination with credentialed requests.
- All client input is validated by Pydantic models (`schemas.py`) with bounds
  on string length, list length, port ranges, and NaN/Inf rejection on feature
  vectors before they ever reach the model.
- Sensor ingestion uses a constant-time API-key comparison (`hmac.compare_digest`)
  to avoid timing attacks.
- JWTs are short-lived (`ACCESS_TOKEN_EXPIRE_MINUTES`) and signed with
  `JWT_SECRET_KEY`, which must be ≥32 chars and never reused across environments.
- Rate limiting (`slowapi`) applies per-IP to both the login endpoint (brute-force
  protection) and ingestion/read endpoints (DoS protection) — tune the limits in
  `.env` for your hardware's capacity.
- Unhandled exceptions are caught globally and returned as a generic 500 with no
  stack trace or internal detail; real errors go to the server log only.
- `/api/docs` and `/redoc` are disabled automatically when `ENVIRONMENT=production`.
- Firebase credentials and `.env` should be excluded from version control
  (add both to `.gitignore`).

## 7. Extending

- **Model**: swap `NeuralNetwork` for a deeper autoencoder, a supervised
  classifier, or an ensemble — the `InferenceEngine` batching logic is
  model-agnostic as long as `forward()`/`reconstruction_error()` conventions
  are preserved (or adapt `_run_batch` for a classifier's softmax output).
- **Auth**: replace `_DEMO_USERS` with a real user table (SQLite/Postgres) and
  add refresh tokens / role management as needed.
- **Persistence**: `PersistenceManager` is the only place that needs to change
  to swap Firestore for another cloud store (e.g. DynamoDB, Supabase).
