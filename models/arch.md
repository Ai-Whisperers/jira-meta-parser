JIRA XML → Validator (cleans schema)
          ↓
variability_features.parquet
          ↓
all-MiniLM-L6-v2 (BGE-M3 if bilingual embeds) → FAISS index → top-K candidates
          ↓
LambdaMART ranks them → clean ordered backlog
          ↓
ColBERTv2 reranks top-50 → final sprint order
