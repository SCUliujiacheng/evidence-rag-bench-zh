"""Evidence-retrieval metrics with explicit denominators."""

from collections.abc import Mapping, Sequence
from math import log2

from evidence_rag_bench.evaluation.cases import EvaluationCase, cases_with_gold


def retrieval_metrics(
    results: Mapping[str, list[str]], cases: Sequence[EvaluationCase], k: int
) -> dict[str, float]:
    """Calculate Recall, reciprocal rank, and binary nDCG for labeled cases."""

    if k < 1:
        raise ValueError("k must be at least one")
    labeled_cases = cases_with_gold(cases)
    if not labeled_cases:
        return {f"recall_at_{k}": 0.0, f"mrr_at_{k}": 0.0, f"ndcg_at_{k}": 0.0}

    recall_total = 0.0
    reciprocal_rank_total = 0.0
    ndcg_total = 0.0
    for case in labeled_cases:
        ranked_ids = results.get(case.case_id, [])[:k]
        gold_ids = set(case.gold_chunk_ids)
        relevant_positions = [
            position
            for position, chunk_id in enumerate(ranked_ids, start=1)
            if chunk_id in gold_ids
        ]
        recall_total += len(set(ranked_ids) & gold_ids) / len(gold_ids)
        if relevant_positions:
            reciprocal_rank_total += 1 / relevant_positions[0]
        dcg = sum(1 / log2(position + 1) for position in relevant_positions)
        ideal_count = min(len(gold_ids), k)
        ideal_dcg = sum(1 / log2(position + 1) for position in range(1, ideal_count + 1))
        ndcg_total += dcg / ideal_dcg if ideal_dcg else 0.0

    denominator = len(labeled_cases)
    return {
        f"recall_at_{k}": recall_total / denominator,
        f"mrr_at_{k}": reciprocal_rank_total / denominator,
        f"ndcg_at_{k}": ndcg_total / denominator,
    }
