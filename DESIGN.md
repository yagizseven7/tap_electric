# Design

This document explains what the service does, the decisions behind it, and what I
learned while building it. Code references point to the files where each idea lives.

**Contents**
1. [The problem](#1-the-problem)
2. [Where the training labels come from](#2-where-the-training-labels-come-from)
3. [Architecture](#3-architecture)
4. [API](#4-api)
5. [Storage](#5-storage)
6. [Model and scan pipeline](#6-model-and-scan-pipeline)
7. [Dataset and training](#7-dataset-and-training)
8. [Validation and promotion](#8-validation-and-promotion)
9. [Testing](#9-testing)
10. [Privacy and security](#10-privacy-and-security)
11. [Running it in production](#11-running-it-in-production)
12. [What building it taught me](#12-what-building-it-taught-me)
13. [Limitations and next steps](#13-limitations-and-next-steps)

---

## 1. The problem

The app's "scan a charger" feature reads the QR sticker on a charger. When the photo is
too dark, blurry, overexposed, or the sticker is faded or damaged, decoding fails and the
driver is stuck. The assignment: a service that collects scan data, trains an
image-to-text model on it, improves that model over time, and validates it.

I treated two questions as the core of the design:

* **Where do correct answers ("labels") come from**, so the model can learn without
  people annotating photos by hand? (Section 2)
* **How do we know a new model is better**, and safe to put in front of drivers?
  (Section 8)

## 2. Where the training labels come from

A model learns from examples with the correct answer. Here they come for free:

| Scan | How we learn the right answer |
|---|---|
| QR decoded | The decoded charger ID is the answer for that photo. |
| QR failed, driver then **typed the ID** printed on the sticker | The typed ID is the answer for the failed photo. |
| QR failed, driver **picked the charger on the map** | The chosen charger's ID is the answer (slightly less reliable: the driver may pick the wrong one). |
| Driver gave up | No answer; still useful as a statistic. |

So the API has two parts: every scan is reported (`POST /v1/scans`), and later the app
reports **what the driver did next** (`PATCH /v1/scans/{id}/outcome`). The second call
turns the scan into a labelled training example, and the failed scans become exactly
the hard examples the model needs most.

Why read the **printed ID**? The QR code and the printed text carry the same charger ID
(an EVSE ID such as `NL*TNM*E12345*1`). When the QR pattern is too damaged to decode,
the printed text is often still readable, and a partially read ID combined with the
driver's GPS position is usually enough to identify the charger.

## 3. Architecture

```
                    ┌───────────────────────── API service (app/) ─────────────────────────┐
 phone app ──scan──▶│ api/scans.py ──▶ storage/repository.py ──▶ PostgreSQL (metadata)     │
           ─outcome▶│                  storage/object_store.py ─▶ S3 (images)              │
           ─resolve▶│ api/resolve.py ─▶ inference/pipeline.py ──▶ production model         │
                    └───────────────────────────────────────────────────────▲──────────────┘
                                                                            │ loads at start-up
                    ┌──────────────── training job (app/training/) ─────────┴──────────────┐
                    │ 1 decide ─▶ 2 dataset ─▶ 3 fine-tune ─▶ 4 register ─▶ 5 validate     │
                    │   job.py    dataset.py    train.py       model_registry  evaluate.py  │
                    │                           (MLflow log)                   promotion.py │
                    └──────────────────────────────────────────────────────────────────────┘
```

* **Two processes.** The API answers drivers in milliseconds on normal servers. The
  training job runs on a schedule on a GPU machine. Separating them means training
  never slows down scanning, and each scales on its own.
* **The model registry connects them.** Training only produces *candidates*; a
  separate validation step decides what becomes *production*; the API loads whatever
  is production. A bad training run cannot reach drivers by itself.
* **Python** throughout: the machine-learning ecosystem (PyTorch, Hugging Face) is
  Python, and FastAPI gives a typed, self-documenting API with little code.
* **Every external system sits behind a small interface** (a `Protocol`), with a real
  implementation and an in-memory one. The assignment asked to write code *as if* the
  databases exist; this way the code is real, and tests run without any infrastructure.

## 4. API

`app/api/`, `app/schemas.py`

| Endpoint | Purpose |
|---|---|
| `POST /v1/scans` | Every scan: image + metadata. |
| `PATCH /v1/scans/{scan_id}/outcome` | What the driver did next: creates the label. |
| `POST /v1/resolve` | Image (+ location) → charger, candidates, or tips. |
| `POST /v1/predict` | The model on its own (for trying it out). |
| `GET /health` | Status and loaded model version. |

### What a scan contains, and why

| Group | Fields | Used for |
|---|---|---|
| Identity | `scan_id`, `session_id`, `captured_at` (with time zone) | de-duplication, grouping retries |
| Device | platform, model, OS and app version | "which phones fail most?" |
| Camera | resolution, exposure, ISO, focus, torch, light sensor (all optional) | explaining failures |
| Location | latitude, longitude, GPS accuracy (optional) | finding nearby chargers |
| Decode result | success, decoded text, decoder, attempts, duration | measuring the current system |
| Consent | `consent_for_training` | GDPR (Section 10) |

### Decisions
* **Multipart upload** (image file + JSON metadata in one request): binary images
  don't belong inside JSON.
* **Validation at the edge.** Pydantic rejects bad data (latitude 500, a "successful"
  scan without decoded text) with HTTP 422 before it reaches storage, so the training
  data stays clean.
* **Idempotent.** The app generates `scan_id`; a retry after a network error returns
  `200 duplicate` instead of storing the scan twice.
* **EXIF metadata is removed** from every image (it can contain exact GPS, device
  serials); the image is also rotated upright first.
* **Image first, then metadata.** If the second write fails, we're left with an unused
  image (harmless), never with a database row pointing to a missing image.
* **The outcome is a separate call** because it happens later, after the driver acts.

## 5. Storage

`app/storage/`

| Store | Holds | Why this one |
|---|---|---|
| **PostgreSQL** | scans, outcomes, chargers, model versions | structured data that is queried and joined ("failed scans with a confirmed charger since last month"); JSONB for camera fields that differ per phone; PostGIS for "chargers within 150 m" |
| **S3** (or MinIO) | scan images | large binary files; cheap, durable; encrypted at rest; date folders (`scans/2026/10/06/…`) make retention deletes easy |
| **MLflow** | training runs: settings, metrics per epoch, model files | the standard open-source experiment tracker |
| **Model registry** (a PostgreSQL table) | which version is candidate / production / archived / rejected, plus lineage (training run, dataset id, base model) | the API reads it at start-up; one transaction guarantees exactly one production model |

The **repository pattern** (`repository.py`) is the only code that talks to the
database. Everything else calls methods like `save_scan()` or `get_labeled_scans()`.
A scan becomes training data only if the driver consented, the charger was found, and
the printed ID is known (`is_trainable`).

## 6. Model and scan pipeline

### Model choice

I compared image-to-text models on Hugging Face:

| Model | Size | For | Against |
|---|---|---|---|
| **microsoft/trocr-base-printed** (chosen) | ~334M parameters | built for printed text; standard fine-tuning recipe; well documented | reads one line, so the photo must be cropped first |
| microsoft/trocr-small-printed | ~62M | same, faster and cheaper | somewhat less accurate |
| microsoft/Florence-2-base | ~230M | can find *and* read text in a whole photo | fine-tuning is more involved |
| Donut, PaliGemma | larger | strong document understanding | heavier to train and serve, overkill for one ID |

TrOCR is small enough to retrain regularly, designed for exactly this kind of text, and
the easiest to explain and fine-tune. The model sits behind a `TextRecognizer`
interface (`inference/model.py`), so moving to Florence-2 later means writing one class.
Each prediction carries a **confidence** (the geometric mean of the token probabilities:
one uncertain character pulls it down, which is right, because one wrong character
means the wrong charger) and the **model version**.

### The pipeline: cheap first, model last

`inference/pipeline.py`

```
image ─▶ 1. decode QR as-is ─────────────── works? ─▶ look up ID ─▶ MATCHED
      ─▶ 2. enhance, decode again ────────── works? ─▶ look up ID ─▶ MATCHED
      ─▶ 3. find text lines ─▶ TrOCR ─▶ compare with chargers near the driver
                ├─ one charger clearly best   ─▶ MATCHED (start charging)
                ├─ several look alike         ─▶ CANDIDATES ("Is it one of these?")
                └─ nothing close              ─▶ NOT_FOUND + tips for the driver
```

* **Enhancement before ML.** Contrast boosting, gamma, denoising and binarisation
  (`enhance.py`) are fast and free. On synthetic damaged stickers they rescued every
  photo the plain decoder missed, including 20 of 20 very dark, noisy photos where the
  plain decoder got 0. The model is only needed when the QR pattern is physically damaged.
* **Matching instead of trusting the text.** The reading is compared with the IDs of
  chargers near the driver (`matching.py`, search radius 150 m – 1 km depending on GPS
  accuracy). A reading with one wrong character still finds its charger.
* **Ask rather than guess.** A charger is started directly only if the reading is very
  similar (≥ 0.85), clearly better than the second-best charger (margin ≥ 0.08), and the
  model is confident enough. Connectors `…*1` and `…*2` of one post differ in one
  character; starting the wrong one is worse than one extra tap. (Section 12 shows the
  default thresholds turned out to be too loose; validation found it.)
* **Help the driver now.** `quality.py` measures brightness, contrast, sharpness and
  glare; when nothing is found the driver gets a specific tip ("It's too dark. Turn on
  the flashlight.") instead of nothing. This helps from day one, before any model exists.
* **The "candidates" answer creates labels.** When the driver taps a suggestion, that
  choice is reported as the outcome, so the pipeline collects exactly the hard
  examples the model is weakest on.
* **Train/serve consistency.** The crop preparation (`prepare_for_model`) is one
  function used by both the pipeline and training, so the model never trains on
  images that look different from what it sees in use.

## 7. Dataset and training

### Building the dataset

`training/dataset.py`

1. **Split by station, with a hash.** All photos of one station (connectors `*1`, `*2`
   …) go to the same split, and the split is computed from a hash of the station ID.
   This prevents *leakage* (testing on a sticker the model practically saw in training),
   and it's stable: when we retrain next month, test stations stay test stations.
2. **Crop the ID line.** We know *what* the sticker says, not *where*. By default the
   current model reads each candidate text line and we keep the one closest to the label
   (`LabelGuidedCropper`); this also filters wrong labels. Without a usable model, a
   model-free check counts letter-shaped blobs and compares with the label length
   (`CharacterCountCropper`).
3. **Measure conditions** (dark, blurry, glare, low contrast) for per-condition results.
4. **Write a manifest** (one JSON line per example) and a **dataset id**, a fingerprint
   of the exact data. Every model records the dataset id it was trained on.
   `dataset_info.json` reports how many examples were kept and dropped *per condition*,
   so losing the hard cases is visible.

### Augmentation

`training/augment.py`. Most scans succeed, so most labels come from easy photos. During
training each crop gets random damage, applied in the physical order it happens: the
sticker (faded, dirty, shadow) → the lens (tilt, blur, motion) → the light (dark,
glare) → the sensor (noise, low resolution) → the phone (JPEG). Each epoch shows a
different damage, so the model learns to read *through* it. Failed scans are also drawn
three times as often (oversampling).

### Fine-tuning

`training/train.py`. A plain PyTorch loop, written out so every step can be explained:
AdamW, learning-rate warm-up and linear decay, gradient clipping, mixed precision on a
GPU. After every epoch the model is measured on the validation split; only the best
epoch is kept, and training stops early when validation stops improving. A run that
never beats its starting point produces nothing. Everything is logged to MLflow.

### The scheduled job

`training/job.py`. It retrains only when worthwhile (2,000 new labelled scans, or 300
new failed ones), starts from the **current production model**, and ends with
validation (Section 8).

### Results (synthetic data)

`try_training.py` trains a tiny model from random weights on 1,200 synthetic scans:
validation character error rate **100% → 5.6%**, **62%** of IDs read exactly right, in
under 3 minutes on a 2-core CPU. This shows the system works end to end; it is
not a measure of real-world accuracy (Section 13).

## 8. Validation and promotion

### The golden test set

`training/golden.py`. A **frozen** set of photos that every model is measured on, so
version 1 and version 7 are directly comparable.

* Only stations from the test split, which training never uses.
* Full photos, not crops: the model is evaluated through the real pipeline, so a bad
  crop counts as a failure, as it would for a driver.
* Images are copied, because normal scan images are deleted after the retention period.
* A `verified` flag marks examples checked by a person (driver labels can be wrong).
* It can't be overwritten by accident; a new golden set gets a new id.

### What is measured

`training/evaluate.py`. Each photo goes through the pipeline with the QR stages switched
off, so the model is tested on every photo.

| Level | Metric | Meaning |
|---|---|---|
| Reading | exact match, character error rate | did it read the ID right? |
| Matching | **resolved rate** | right charger started, or suggested **first** |
| | **GPS baseline** | how often the nearest charger is simply right, with no model |
| | **wrong-charger rate** | started the wrong charger: the most dangerous error |
| System | **rescue rate** | of the scans where the QR code failed, the share now resolved: the business value |

Plus per-condition **slices**, **calibration** (does "90% confident" mean right 90% of
the time? The report suggests the confidence threshold above which readings are 99%
right), and **latency** (p50 / p95).

### The promotion gate

`training/promotion.py`. A candidate replaces the production model only if it passes
**every** check; each check reports its numbers.

1. **Minimum quality:** reads ≥ 80% of IDs exactly, resolves ≥ 85% of photos, and
   beats the GPS-only baseline by ≥ 10 points.
2. **Safety:** at most 1% wrong-charger starts, and at most 0.5 points more than the
   current model.
3. **Clearly better than the current model:** checked with a *paired bootstrap*. Both
   models are compared photo by photo, the test photos are re-drawn at random 2,000
   times, and the 95% lower bound of the gain must be above zero. A gain that luck in
   the choice of test photos could explain doesn't count.
4. **No slice worse:** no condition (dark, blurry, …) with enough examples drops more
   than 3 points, even if the average improves.
5. **Fast enough:** 95% of scans within the latency budget.

The current model is re-evaluated each time with the same code and the same golden set,
so the comparison stays fair. Reports are saved per candidate (`reports/`).

### After offline validation

* **Shadow mode** first: the new model reads every real failed scan next to the
  current one, but drivers only see the current model's answer. If the shadow results
  match the offline ones for about a week, it takes over.
* **Monitoring**: the manual-entry rate after failed scans, the wrong-charger reports
  from support, and the confidence distribution (a shift means the photos changed).
* **Rollback**: mark the previous version as production and restart the API.

### Results (synthetic data)

`try_validation.py`:

| Round | Exact | Resolved | GPS alone | Wrong charger | Decision |
|---|---|---|---|---|---|
| Weak model (8 epochs) | 0% | 6% | 49% | 0% | rejected: 3 checks failed |
| Properly trained (50 epochs) | 76.7% | 98.3% | 49% | 0% | promoted |
| More training from the champion | – | – | – | – | not even registered: didn't beat its starting point |

The demo uses lower quality bars than production (20% exact instead of 80%), because
its tiny model learns from zero in minutes. I chose that demo bar after seeing the
tiny model's score; in real use the bars must be fixed *before* any candidate is evaluated.

## 9. Testing

115 tests (`tests/`), covering 97% of `app/`; GitHub Actions runs them on every push.

| Kind | Example |
|---|---|
| Unit | the paired bootstrap rejects a +1 gain on 20 photos |
| Contract | every storage test runs against the in-memory **and** the SQL repository, proving the fakes behave like the real thing |
| Integration | a fake model evaluated on real generated sticker images |
| End-to-end | scans and outcomes through the HTTP API → training → validation → promotion → a restarted API answers with the new model |

Slow tests are marked (`pytest -m "not slow"` runs 110 tests in ~20 s). Tests use a fake
model or a tiny TrOCR built on the spot, so they need no downloads.

## 10. Privacy and security

Scan photos plus location are personal data under the GDPR.

* **Consent:** only scans with `consent_for_training` are used for training.
* **Data minimisation:** EXIF metadata is stripped; no user id is stored with a scan.
* **Encryption at rest** for images in S3.
* **Retention:** images are stored in date folders so they can be deleted after a fixed
  period (e.g. 12 months). Golden-set images are kept longer, on the basis of the
  consent, and reviewed by a person.
* **Not implemented yet:** authentication of the app (e.g. signed tokens) and rate
  limiting on the endpoints (Section 13).

## 11. Running it in production

* **API:** stateless containers behind a load balancer; scale on request volume.
  The model loads once at start-up; inference runs off the main thread.
* **Training job:** a scheduled job (Kubernetes CronJob, Airflow or a cloud scheduler)
  on a GPU machine.
* **Infrastructure:** PostgreSQL with PostGIS, an S3 bucket, an MLflow server whose
  files live in S3. All configured with environment variables (see README).

## 12. What building it taught me

I tested each part as I built it, and several problems only showed up because I
measured instead of assuming. Each one led to a fix, a check or a test.

1. **Crops were often wrong at first:** pieces of QR code, empty background, an ID
   missing its first letters. *Fix:* merge text pieces on the same row, and verify a
   crop by counting letter-shaped blobs against the label.
2. **The dataset silently dropped the hard photos.** Only 1 of 33 dark and 0 of 11
   blurry photos survived cropping, because local contrast boosting turned sensor noise
   into fake edges. *Fix:* denoise before detecting text (dark: 33/33), and count letters
   at several darkness levels (blurry: 8/11). This also fixed the live pipeline, which
   uses the same detector. The dataset report now shows drops per condition.
3. **A useless model counted as "improved".** A model that wrote the same "average" ID
   for every sticker still beat a model that read nothing. *Fix:* absolute minimum
   quality bars in the gate.
4. **A metric flattered a model that couldn't read.** It scored 97.6% "resolved",
   because the right charger appeared *somewhere* in the suggestion list, which GPS alone
   would achieve. *Fix:* count only "started" or "suggested first", and require beating a
   GPS-only baseline.
5. **The model memorised instead of reading.** With only ~130 different station numbers,
   training loss fell to 0.08 while validation error stayed at 40%. With ~480 different
   numbers it learned to read. **What matters is the number of different IDs, not the
   number of photos.** Rare operators and new ID formats are therefore a real risk.
6. **The pipeline could start the wrong connector.** A misread `*1` for `*2` gives a
   similarity margin of 0.091, just above the 0.08 rule. The evaluation measures this,
   the gate blocks such a model, and a test shows that a calibrated confidence threshold
   prevents it.
7. **My synthetic data was too easy in two ways:** chargers were never neighbours, and
   the fake GPS was perfect (a 100% "GPS baseline"). *Fix:* stations with 1–4 connectors
   and realistic 5–20 m GPS error, after which GPS alone found the right charger 49% of
   the time.
8. **The model is over-confident:** 94% confident on wrong readings. The calibration
   report makes this visible and turns it into a concrete threshold.
9. **Training details for a model starting from zero:** early stopping cut it off
   during an initial plateau, and square image patches never learned to read; thin
   vertical patches did. (A pre-trained TrOCR has neither problem.)

## 13. Limitations and next steps

**Limitations**
* All model results come from **synthetic** stickers. Real photos add perspective,
  reflections, dirt and real fonts; thresholds tuned here must be re-tuned on real data.
* The real TrOCR was not fine-tuned for this submission (my development environment
  could not download it); the code supports it (`python try_training.py --real`).
* Text-line detection is a hand-written heuristic; the hardest blurry photos are still
  sometimes dropped from training.
* No authentication or rate limiting on the API; no database migrations (Alembic).
* QR codes containing something other than an EVSE ID (e.g. an operator's URL) are not
  looked up.
* Shadow mode and monitoring are designed, not implemented; the golden set's
  `verified` flag has no review tool yet.

**Next steps, in order**
1. Collect real scans and outcomes (the API is ready), and build a human-verified
   golden set from them.
2. Fine-tune the real TrOCR on them; set the pipeline's `min_ocr_confidence` from the
   calibration report.
3. Mix in rendered stickers with **random** IDs during training, against memorisation
   and for unseen operators.
4. Replace the text-line heuristic with a trained detector (e.g. Florence-2).
5. Implement shadow mode, monitoring dashboards, and authentication.
