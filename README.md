# QR Scan Service

A service that helps EV drivers when the **"scan a charger"** feature fails, and that
uses every scan to get better at it.

When a driver points their phone at a charger's QR sticker, the scan sometimes fails:
the photo is too dark, blurry, there is glare, or the sticker is faded. This service:

1. **collects** every scan (image + useful context) through an API,
2. **learns from what happens next**: when a scan fails, the driver usually still finds
   the charger by typing the printed ID or picking it on the map. That choice is the
   correct answer for the failed photo, so labels come for free,
3. **reads the printed charger ID** when the QR code can't be decoded, using an
   open-source image-to-text model ([TrOCR](https://huggingface.co/microsoft/trocr-base-printed)),
   and matches it against chargers near the driver,
4. **retrains** that model on the collected scans, and
5. **validates** every new model on a frozen test set before it may replace the current one.

The design decisions and their reasons are in **[DESIGN.md](DESIGN.md)**.
How I worked with an AI coding assistant is in **[CHAT.md](CHAT.md)**.

```
 phone app ─── POST /v1/scans ───────────▶ ┌──────────────┐   images  ──▶ S3
           ─── PATCH /v1/scans/{id}/outcome│   this API   │   metadata ──▶ PostgreSQL
           ─── POST /v1/resolve ──────────▶│              │◀── production model
                                           └──────────────┘          ▲
                                                  │ labelled scans    │ promote
                                                  ▼                   │
                                    training job (scheduled, GPU) ────┘
                                    dataset → fine-tune → validate on golden set
```

---

## Quick start

Requires **Python 3.11 or newer**.

```bash
git clone <this repository>
cd qr-scan-service
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # the PyTorch download is large; this takes a while
pytest                             # 115 tests, about a minute
```

No database, S3 or GPU is needed for any of this: without configuration the service
uses in-memory storage, and the tests use a tiny model built on the spot.

### Run the API

```bash
python make_test_stickers.py       # writes test sticker photos to sample_data/stickers/
CHARGERS_CSV=sample_data/chargers.csv uvicorn app.main:app --reload
```

On Windows (PowerShell): `$env:CHARGERS_CSV="sample_data/chargers.csv"; uvicorn app.main:app --reload`

Open **http://localhost:8000/docs** for interactive documentation where you can try
every endpoint. Or from a second terminal:

```bash
# 1. The app reports a failed scan
curl -X POST localhost:8000/v1/scans \
  -F 'metadata={"scan_id":"3f1c2a9e-1b2c-4d5e-8f90-123456789abc","session_id":"7a8b9c0d-1e2f-4a3b-9c4d-abcdef123456","captured_at":"2026-10-06T14:00:00+02:00","device":{"platform":"ios","device_model":"iPhone 15","os_version":"17.5","app_version":"4.12.0"},"camera":{"image_width":460,"image_height":520,"ambient_lux":3.0},"location":{"latitude":52.37,"longitude":4.89,"accuracy_m":8},"decode":{"success":false,"duration_ms":6000,"attempts":40},"consent_for_training":true}' \
  -F 'image=@sample_data/stickers/very_dark.jpg;type=image/jpeg'

# 2. Later: the driver found the charger by typing its ID -> this scan becomes a training example
curl -X PATCH localhost:8000/v1/scans/3f1c2a9e-1b2c-4d5e-8f90-123456789abc/outcome \
  -H 'Content-Type: application/json' \
  -d '{"method":"manual_entry","charger_id":"CH-1","evse_id":"NL*TNM*E12345*1","resolved_at":"2026-10-06T14:01:00+02:00"}'

# 3. Ask the service to find the charger on a hard photo
curl -X POST localhost:8000/v1/resolve \
  -F 'image=@sample_data/stickers/very_dark.jpg;type=image/jpeg' -F latitude=52.37 -F longitude=4.89
```

The last call answers `"status": "matched", "charger_id": "CH-1"`: the phone's QR decoder
gave up on this dark photo, but the service's enhanced decoding rescued it
(`"stage": "qr_smooth"`). To also use the model for stickers whose QR code is physically
damaged, start the server with `LOAD_MODEL=true` (downloads TrOCR, ~1.3 GB, the first time).

### Watch the model learn and the promotion gate decide

```bash
python try_training.py      # trains a small model on synthetic scans (~5 min on a laptop)
python try_validation.py    # the gate rejects a weak model and promotes a good one (~8 min)
mlflow ui --backend-store-uri sqlite:///mlflow.db    # training charts at http://localhost:5000
```

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /v1/scans` | Every scan: the image plus metadata (device, camera, location, decode result, consent). Idempotent: a retried `scan_id` is not stored twice. EXIF metadata is stripped from the image. |
| `PATCH /v1/scans/{scan_id}/outcome` | How the driver finally found the charger (`qr`, `manual_entry`, `map_selection`, `abandoned`). Turns the scan into a labelled training example. |
| `POST /v1/resolve` | Image (+ location) in, charger out: `matched`, `candidates` ("Is it one of these?") or `not_found` with tips for the driver ("It's too dark. Turn on the flashlight."). |
| `POST /v1/predict` | The image-to-text model on its own. |
| `GET /health` | Status and the version of the loaded model. |

---

## Project layout

```
app/
  main.py, config.py         start-up and settings
  schemas.py                 what the API accepts and returns
  api/                       the endpoints
  storage/                   PostgreSQL tables, S3 images, chargers, model registry
  inference/                 the pipeline: QR decoding → enhancement → model → matching
  training/                  dataset, augmentation, training, golden set, evaluation, promotion, job
tests/                       115 tests (unit, contract, integration, end-to-end)
try_*.py                     demo scripts to run by hand
sample_data/chargers.csv     test chargers in Amsterdam
.github/workflows/tests.yml  runs the tests on every push
```

---

## Configuration

All settings are environment variables, so the same code runs on a laptop, in tests and
in production.

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | *(none: in-memory)* | e.g. `postgresql+psycopg://user:pass@host:5432/scans` (also `pip install "psycopg[binary]"`) |
| `S3_BUCKET` | *(none: in-memory)* | bucket for scan images |
| `S3_ENDPOINT_URL` | *(AWS)* | set to use MinIO or another S3-compatible store |
| `LOAD_MODEL` | `false` | load the image-to-text model at start-up |
| `MODEL_NAME` | `microsoft/trocr-base-printed` | model used when the registry has no production model yet |
| `CHARGERS_CSV` | *(none)* | load chargers from a CSV instead of the database (local testing) |
| `MAX_IMAGE_BYTES` | `10485760` | upload limit (10 MB) |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | where the training job logs its runs |

---

## Training in production

The training job runs on a schedule (e.g. nightly) on a GPU machine, separately from the API:

```bash
DATABASE_URL=... S3_BUCKET=... python -m app.training.job --golden-dir golden/v1
```

Each run: checks whether enough new data has arrived → builds the dataset → fine-tunes
the current production model → evaluates the result and the current model on the golden
test set → promotes the new model only if it passes every check. The API loads the
production model at start-up; rolling back means marking the previous version as
production again. Details in [DESIGN.md](DESIGN.md#8-validation-and-promotion).

---

## Tests

```bash
pytest -m "not slow"     # 110 fast tests, ~20 seconds: run these while coding
pytest                   # all 115, including training and the full loop
pytest --cov             # with a coverage report (97% of app/)
```

---

## Status and limitations

This is an interview assignment: the infrastructure (PostgreSQL, S3, MLflow server) is
written against but not deployed, and all model results so far come from **synthetic**
sticker photos, not real scans. The most important open points are listed in
[DESIGN.md → Limitations and next steps](DESIGN.md#13-limitations-and-next-steps).
