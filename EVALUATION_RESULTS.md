# Evaluation Results

## CodeRankEmbed on SWE-Bench-Lite

### Model Information
- **Model**: `nomic-ai/CodeRankEmbed`
- **Parameters**: 137M parameter bi-encoder
- **Context Length**: 8192 tokens
- **Query Prefix**: `"Represent this query for searching relevant code"`

### Dataset
- **Dataset**: SWE-Bench-Lite (function-level)
- **Total Instances**: 274
- **Evaluation Type**: Software issue localization

### Performance Metrics

| Metric | @1 | @3 | @5 | @10 | @100 | @1000 |
|--------|----|----|----|----- |------|-------|
| **NDCG** | 25.91% | 36.18% | 40.63% | 43.15% | 48.06% | 49.61% |
| **MRR** | 26.28% | 34.98% | 37.11% | 38.05% | 39.00% | 39.04% |
| **Recall** | 23.97% | 43.69% | 53.99% | 61.59% | 83.70% | 95.62% |
| **Precision** | 25.91% | 16.30% | 12.26% | 7.01% | 0.97% | 0.11% |

### Key Results
- **Top-1 Accuracy**: 25.91% of instances had the correct function ranked first
- **Top-5 Recall**: 53.99% of relevant functions found within top 5 results
- **Top-10 Recall**: 61.59% of relevant functions found within top 10 results
- **Average Processing Time**: ~105 seconds per instance
- **Total Evaluation Time**: ~7.5 hours

### Technical Implementation
- Successfully integrated CodeRankEmbed into the SweRank evaluation framework
- Modified `eval_beir_sbert_canonical.py` to support the model's specific query prefix requirements
- Processed all 274 instances without failures
- Results saved to: `./results/model=CodeRankEmbed_dataset=swe-bench-lite_split=test_level=function_evalmode=default_results.json`

### Command Used
```bash
bash script/run_retriever.sh nomic-ai/CodeRankEmbed CodeRankEmbed /path/to/datasets swe-bench-lite
```

---

*Evaluation completed on: June 29, 2025*
