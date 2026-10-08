# Working with an AI coding assistant

The assignment asked to use a coding assistant and share the conversation. I used
**Claude** (Anthropic), in one conversation spread over three days (6–8 October 2026).

**Full conversation:** ✏️ *https://claude.ai/share/a6ded4db-4760-4a34-8336-fdb8d6196530*

This file is a guide to that conversation: every prompt I sent (word for word), what
came back, and the decisions and problems along the way. The full answers, including all
code and test output, are in the shared conversation.

---

## How I worked

* **Plan first, then one step at a time.** I first asked for a complete plan (design
  decisions, setup, build order). Then I asked for one or two steps at a time, so each
  piece was built, tested and explained before the next one depended on it.
* **Explanations, not just code.** I asked for every step to be explained in detail, so
  I can explain and defend each decision myself.
* **Everything was run, not only written.** The assistant ran the code and the tests in
  its own environment after each step, and reported problems it found, including
  problems in its own earlier code. Section "Problems found along the way" lists them.

---

## Conversation log

### 1. The plan — 6 Oct, 14:53

> I have a task below. Which steps should I follow to complete it? I am a complete
> beginner so explain every step in detail.
>
> *(followed by the full assignment text)*

**Answer:** a four-phase plan.
* *Phase 0, the problem:* the key insight is that labels come for free. Successful
  scans label themselves, and after a failed scan the driver's manual entry or map
  choice is the correct answer for that photo.
* *Phase 1, decisions:* Python with FastAPI, PyTorch and Hugging Face; PostgreSQL for
  metadata, S3 for images, MLflow for experiments; TrOCR as the model (compared with
  Florence-2, Donut, PaliGemma); a pipeline that tries cheap QR decoding before the
  model; GDPR considerations.
* *Phase 2–4:* setting up Python, a virtual environment and Git; the build order; the
  validation strategy; documentation and submission.

### 2. Step 5: schemas — 6 Oct, 16:03

> In phase 3, step 5, How can I implement this?

**Answer:** `app/schemas.py`: Pydantic classes for everything the API accepts and
returns, with field rules (latitude between −90 and 90) and cross-field rules (a
"successful" scan must include the decoded text; an outcome must name a charger unless
the driver gave up). Tested with valid and invalid data.

### 3. Step 6: storage — 6 Oct, 16:47

> In phase 3, step6, How can I implement this?

**Answer:** `app/storage/`: PostgreSQL tables with SQLAlchemy, S3 image storage, and the
repository pattern, each with an in-memory version for tests. Rules: duplicate retries
aren't stored twice; only consented, resolved scans become training data.

### 4. Steps 7 and 8: API and model — 6 Oct, 17:00

> In phase 3, step7 and step8, How can I implement these?

**Answer:** the FastAPI app (`main.py`, `api/`) with an app factory, dependency
injection, idempotent scan upload, EXIF removal, image-before-metadata ordering; and the
TrOCR wrapper (`inference/model.py`) with beam search, a confidence score and a model
version on every prediction. 14 tests.

### 5. Step 9: the scan pipeline — 6 Oct, 18:24

> In phase 3, step9, How can I implement this?

**Answer:** `inference/pipeline.py` and helpers: QR decoding → image enhancement →
model → matching against chargers near the driver, with "ask rather than guess" rules
and tips for the driver. An experiment on damaged synthetic stickers showed enhancement
alone rescues the QR code in every tested condition (0/20 → 20/20 for very dark, noisy
photos). 41 tests.

### 6. Steps 10 and 11: dataset and training — 7 Oct, 16:38

> In phase 3, step10 and step11, How can I implement these?

**Answer:** the dataset builder (split by station with a hash, cropping, manifest and
dataset id), physically ordered augmentation, synthetic sticker generation, a plain
PyTorch training loop with early stopping, MLflow tracking, a model registry, and the
scheduled training job. A demo trained a tiny model from scratch: character error rate
100% → 10.8%. 70 tests.

### 7. Steps 12 and 13: validation and tests — 8 Oct, 14:27

> In phase 3, step12 and step13, How can I implement these?

**Answer:** the frozen golden test set, evaluation at three levels (reading, matching,
business value) with slices, calibration and latency, and a promotion gate with five
checks including a paired bootstrap. The test suite was reorganised (shared helpers and
fixtures, slow-test marker, coverage, contract tests, an end-to-end test, GitHub
Actions). 115 tests, 97% coverage.

### 8. Overview — 8 Oct, 16:24

> Now, provide overall folder hierarchy with all relevant files. I need to modify

**Answer:** the full folder tree with a description of every file, the settings most
likely to need changing, and the whole project as one zip file (all tests passing).

### 9. Documentation — 8 Oct, 16:50

> Yes, please. Also explain every step in detail.

**Answer:** `README.md`, `DESIGN.md` and this file, with every README command checked
against the running service, plus a step-by-step guide to finishing and submitting.

---

## Problems found along the way

The most useful part of the collaboration was that every step was **measured**, and
several problems only appeared because of that. Each became a fix, a check or a test.
Details are in [DESIGN.md, section 12](DESIGN.md#12-what-building-it-taught-me).

| When | Problem | What was done |
|---|---|---|
| Step 9 | Digitally darkened test photos were too easy to be realistic | Added sensor noise and JPEG compression to test images |
| Step 9 | A URL in a QR code was parsed as an ID starting with "HTTPS" | Fixed the ID parser; a test caught it |
| Step 10 | Crops were often wrong (QR fragments, empty background, missing first letters) | Merge text pieces per row; verify crops by counting letters |
| Step 10 | Dark and blurry photos were silently dropped from the dataset (1/33, 0/11) | Denoise before text detection (33/33, 8/11); this also fixed the live pipeline |
| Step 11 | A model that wrote the same "average" ID counted as improved | Absolute quality bars in the promotion gate |
| Step 11 | A model trained from zero didn't learn to read | Vertical image patches for the tiny demo model |
| Step 12 | A model that couldn't read scored 97.6% "resolved" | Stricter metric plus a GPS-only baseline |
| Step 12 | The model memorised ~130 IDs instead of reading | More different IDs in the data; random-ID stickers as a next step |
| Step 12 | Synthetic GPS was perfect, so GPS alone looked 100% right | Realistic GPS error (5–20 m) |
| Step 12 | Default thresholds could start the wrong connector (`*1` for `*2`) | Detected by the evaluation; fixed with a calibrated confidence threshold |
| Step 13 | The SQL repository was never tested (coverage 66%) | Contract tests against both implementations (100%) |

One judgement call worth noting: the quality bar of the *demo* promotion gate was set
after seeing the tiny model's score. That's acceptable for demonstrating the mechanism,
but in real use the bars must be fixed before any model is evaluated.
