Here’s the sweet spot for “dirty JIRA” → clean, ordered backlog:

## Recommended model (2-stage, fast, small, reliable)

1. **Embedding (semantic)** — *Sentence-BERT mini*

   * **Model**: `all-MiniLM-L6-v2`
   * **Why**: robust sentence/paragraph embeddings for mixed English summaries + descriptions; tiny VRAM; great recall.

2. **Learning-to-Rank (LTR)** — *LambdaMART (LightGBM)*

   * **Objective**: `lambdarank` (pairwise).
   * **Why**: Orders tickets by “actionability/priority” using both semantics and backbone flags; deterministic, stable, easy to tune.

---

## Features (keep it lean)

* **Semantic**: embedding[64/384] from summary + description (concat or mean).
* **Backbone hygiene**: `required_ok`, `key_format_ok`, `dates_ok`, `unique_key_ok`.
* **Dirty flags**: `flag_missing_ac`, `assignee_empty`, `storypoints_empty`, `summary_len`, `desc_len`.
* **Context anchors**: one-hot (`type`, `status`, `priority`).
* **Light structure**: `label_count`, `component_count`, `customfield_count`, `link_count`.

---

## Weak labels (so you can train today)

Define a target **priority score** to supervise LTR without manual labeling:
[
\text{y} = 2\cdot\mathbb{1}[\text{priority}\in{High,Critical}] +
1\cdot\mathbb{1}[\text{status}\in{\text{Selected for Dev},\text{In Progress}}] +
1\cdot\mathbb{1}[\text{type}\in{\text{Bug},\text{Story}}] -
2\cdot\mathbb{1}[\text{flag_missing_ac}] -
1\cdot\mathbb{1}[\text{assignee_empty or storypoints_empty}]
]
Then **group by epic/sprint/project** for LTR “query_id”.

---

## Training recipe (quick)

* **Embedder**: freeze (or fine-tune 1–2 epochs; batch 64; lr 1e-5).
* **LightGBM LambdaMART**:

  * `num_leaves=31`, `learning_rate=0.06`, `n_estimators=500`, `min_data_in_leaf=20`, `feature_fraction=0.8`.
  * Metric: `ndcg@10` (also watch `ndcg@20`).
* **Eval**: `NDCG@k`, `Kendall τ` vs. your current heuristic order, and “human approval @ top-20”.

---

## Inference flow (deterministic & fast)

1. Validate XML → generate **variability_features.parquet** (we built this).
2. Compute embeddings (MiniLM or your 64-d model).
3. Score with LambdaMART → **rank** inside each epic/sprint/project.
4. Post-rules (guardrails):

   * If `flag_missing_ac==1` ⇒ push below any ticket with AC.
   * If `assignee_empty or storypoints_empty` ⇒ push behind fully specified siblings.

**Output**: an ordered CSV (`rank`, `key`, `score`, `why_top` with top contributing features).