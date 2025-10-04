# JIRA → Clean Ordered Backlog (v1) — **Production-Ready Blueprint**

> Purpose: take “dirty, chaotic” JIRA XML exports and produce a **deterministic, explainable, high-quality order** for planning—without overcomplicating the stack.
> Core idea: **separate schema hygiene from semantics**, then **rank with a learning-to-rank model**, and (optionally) **polish** the top of the list with a neural re-ranker. 

---

## 0) **Executive Summary**

* **Ingest**: raw `JIRA.xml`
* **Validate**: strict schema backbone (keys, dates, required fields)
* **Extract**: variability-only features (summary, description, low-cardinality counts)
* **Embed**: `all-MiniLM-L6-v2` (384-D)
* **Index**: FAISS (IVF-PQ) → fast top-K candidate retrieval
* **Rank**: LightGBM **LambdaMART** (grouped by epic/sprint) → **ordered backlog**
* **(Optional)**: **ColBERT-v2** re-rank for top-50 “final polish”
* **Outputs**: `clean_backlog.csv` (rank, score, explanations), drift & quality dashboards. 

---

## 1) **Data Contracts & Artifacts**

### 1.1 Inputs

* `JIRA.xml` (RSS or `<issues/issue>` form)
* (Optional) historical `clean_backlog.csv` for continual evaluation

### 1.2 Intermediate

* `backbone_report.csv`, `backbone_summary.json` — schema validator outputs
* `variability_features.parquet` — ML-ready table (**only** variability, no meta-structure) 

### 1.3 Outputs

* `embeddings.parquet` (N × 384 for MiniLM)
* `faiss_index.ivf` (+ quantizer files)
* `ltr_model.txt` (LightGBM), `feature_map.json`
* `clean_backlog.csv` with:

  * `rank, key, score, epic, sprint, status, priority`
  * `why_top`: top contributing features (SHAP/feature gain)
  * `guardrails_applied`: e.g., “missing AC pushed down”

---

## 2) **Pipeline (single pass)**

```text
JIRA.xml
  ↓ (A) Validator (O(n))
backbone_report.csv, backbone_summary.json
  ↓ (B) Variability Extractor (O(n))
variability_features.parquet  [summary_txt, description_txt, counts, anchors]
  ↓ (C) Embeddings: all-MiniLM-L6-v2 (384-D)
embeddings.parquet
  ↓ (D) FAISS IVF-PQ → top-K candidates per query group
candidates.parquet
  ↓ (E) LightGBM LambdaMART (lambdarank) → final order
clean_backlog.csv
  ↓ (F) (Optional) ColBERT-v2 re-rank top-50
clean_backlog.csv (polished)
```

**Design rationale:** simple, modular, and explainable; no O(n²) pairwise similarities; compute fits on a single GPU/CPU workstation. 

---

## 3) **Module Specs**

### (A) Backbone Validator

* **Checks:** `key, summary, type, status, priority, created, updated` presence; key regex `^[A-Z][A-Z0-9_]+-\d+$`; dates parseable; link references valid.
* **Fail-fast:** invalid records logged and quarantined—not fed to ML.
* **Why:** forces **constant schema** compliance so the model learns only **variability**, not meta-structure. 

### (B) Variability Feature Extractor

* **Text:** `summary_txt`, `description_txt` (HTML-stripped, unescaped)
* **Counts:** `label_count`, `component_count`, `customfield_count`, `link_count`
* **Anchors:** categorical `type, status, priority` (for one-hot)
* **Note:** keep it **lean**; the ranker’s job is ordering, not heavy NLP. 

### (C) Embeddings — **all-MiniLM-L6-v2**

* **Pooling:** mean-pool on `[summary_txt + description_txt]`
* **Dim:** 384-D; **normalize L2** for cosine sim.
* **BGE-M3** reserved as future multilingual drop-in (not needed now). 

### (D) FAISS Index

* **Index type:** IVF-PQ (coarse quantizer nlist=4096, m=16 (PQ), nprobe=16)
* **Recall/latency knobs:** tune `nprobe` for trade-off; store on GPU if fits.
* **Query grouping:** retrieve **top-K per group** (epic/sprint) to bound search space for ranking.

### (E) Learning-to-Rank — **LambdaMART (LightGBM)**

* **Objective:** `lambdarank`, **group** = `epic` or `sprint`
* **Features** (minimal but strong):

  * **Semantic**: top-K sim scores, or dot(emb_i, centroid_group)
  * **Hygiene**: from validator (e.g., `required_ok`, `key_format_ok`, `dates_ok`)
  * **Dirty flags**: `flag_missing_ac`, `assignee_empty`, `storypoints_empty`, lengths
  * **Anchors**: one-hot `type, status, priority`
  * **Structure**: counts (labels/components/customfields/links)
    *(Feature set mirrors the “sweet spot” recipe.)* 
* **Params (good defaults):**
  `num_leaves=31, learning_rate=0.06, n_estimators=500, min_data_in_leaf=20, feature_fraction=0.8`
  **Metrics:** `ndcg@10`, `ndcg@20`, **Kendall τ** vs. current manual order. 
* **Weak-label target (if no labels yet):**
  Weighted rule combining priority, status, type, and penalty for missing AC/assignee/points; then learn pairwise preferences *within* each group. 

### (F) Optional Re-rank — **ColBERT-v2 (top-50)**

* **Scope:** only the final **top-50** per group; late-interaction boosts nuance (synonyms, phrasing).
* **Integration:** replace rank score = `0.7 * LTR_score + 0.3 * ColBERT_score` (tunable).
* **When to enable:** if PMs still flag edge-cases in top-10 after LTR.

---

## 4) **Guardrails (Business Rules After LTR)**

* If `flag_missing_ac == 1` → demote below any sibling with AC.
* If `assignee_empty or storypoints_empty == 1` → demote behind fully specified siblings.
* If `status ∈ {Blocked, On Hold}` → cap rank ≤ 10th percentile of its group.
* If `type == Bug` and `priority ∈ {High, Critical}` → minimum rank uplift (tie-breaker).

These ensure **operational sanity** even when models are uncertain. 

---

## 5) **Training & Inference Routines**

### 5.1 Training (daily or on change)

1. Validate XML → write hygiene stats.
2. Extract variability features.
3. Compute/update embeddings; rebuild FAISS (or add vectors to IVF).
4. Prepare weak labels (or real labels if available).
5. Train LightGBM; log metrics (`ndcg@10`, τ).
6. Persist artifacts (`ltr_model.txt`, `feature_map.json`).

### 5.2 Inference (on demand)

1. New XML → validator pass (drop invalid).
2. Feature & embedding compute for deltas.
3. FAISS retrieve top-K per group.
4. LTR score → `clean_backlog.csv`.
5. (Optional) ColBERT re-rank top-50; overwrite ranks.

---

## 6) **Quality & Observability**

* **Dashboards:**

  * Hygiene coverage (AC present, owner, estimate).
  * NDCG@K per sprint/epic; drift of embedding centroid per group.
  * Post-rules hit-rates (how many demotions happened, why).
* **Human-in-the-loop:**

  * Quick UI: “approve/reject top-20” → writes feedback to `labels.parquet` for future LTR.
* **Ablations (monthly):**

  * Remove one feature bucket at a time → confirm it actually contributes.
* **Canary:**

  * Keep 10% sprints on “old heuristic order” to compare velocity & revert if needed.

---

## 7) **Performance Targets**

* **Throughput**: 10–50k tickets end-to-end in < 5 min on 1× RTX-3050 + CPU.
* **Recall** (FAISS top-K): ≥ 0.95 vs brute-force on a 2k sample.
* **NDCG@10**: ≥ 0.85 on weak labels; **PM approval@20** ≥ 80%.
* **Stability**: Kendall τ ≥ 0.6 between runs on same snapshot.

---

## 8) **Security & Compliance**

* Process XML **locally**; no external calls.
* Strip HTML from descriptions; redact credentials/URLs if patterns match.
* Version `JIRA.xml`, `clean_backlog.csv`, and models; pin hash to audit.
* If ColBERT-v2 is used, cache token embeddings; do not ship raw text.

---

## 9) **Failure Modes & Mitigations**

* **Validator rejects many issues** → block release; report exact missing fields.
* **Embedding drift** (sudden language/content shift) → re-compute FAISS & retrain LTR.
* **Rank instability** → raise `min_data_in_leaf`, reduce `num_leaves`, add guardrails weight.
* **Top-K misses** → increase FAISS `nprobe`; raise K from 100 → 200.

---

## 10) **Minimal Config (copy/paste)**

**FAISS (IVF-PQ):**

* `nlist=4096` (coarse clusters), `m=16` (PQ sub-vectors), `nprobe=16` (query-time scan)

**LightGBM (LambdaMART):**

* `objective=lambdarank`, `metric=ndcg`, `label_gain=[0,1,3,7]`
* `num_leaves=31`, `learning_rate=0.06`, `n_estimators=500`, `feature_fraction=0.8`, `min_data_in_leaf=20` 

**ColBERT-v2 (optional):**

* Apply only to **top-50**; blend `final = 0.7 * ltr + 0.3 * colbert`.

---

## 11) **Why This Works (and stays simple)**

* Clean separation of concerns (schema → semantics → rank → polish).
* No bespoke giant NN; uses **battle-tested** MiniLM + LightGBM; ColBERT only if needed.
* Deterministic, interpretable, and **fast**; easy to maintain and extend later (e.g., multilingual). 

---

### Appendix A — Feature List (concise)

* **Semantic**: `sim_top1..topK` or `emb_dot_to_group_centroid`
* **Hygiene**: `required_ok, key_format_ok, dates_ok, unique_key_ok`
* **Dirty flags**: `flag_missing_ac, assignee_empty, storypoints_empty, summary_len, desc_len`
* **Anchors**: one-hot `type, status, priority`
* **Structure**: `label_count, component_count, customfield_count, link_count`
  *(These mirror the proven set from our “sweet spot” recipe.)* 

---

### Appendix B — Canonical Flow Reference

The final flow you approved (kept verbatim here as a contract):

**JIRA XML → Validator (cleans schema) → variability_features.parquet → all-MiniLM-L6-v2 → FAISS index → top-K → LambdaMART → clean ordered backlog → ColBERT-v2 top-50.** 

---


Medical References:
1. None — DOI: file_00000000308861f68461f4ab48670191
2. None — DOI: file_00000000674061f6825516cb921e4a63
3. None — DOI: file_000000005f6461f6938d3ddfa263f98b