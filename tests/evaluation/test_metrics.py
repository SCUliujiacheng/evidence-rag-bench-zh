from evidence_rag_bench.evaluation.cases import EvaluationCase
from evidence_rag_bench.evaluation.metrics import retrieval_metrics


def test_recall_at_k_counts_gold_evidence() -> None:
    cases = [
        EvaluationCase(
            case_id="q1",
            split="dev",
            question="first",
            answerability="answerable",
            gold_chunk_ids=["a:0000"],
            reference_answer="first",
            notes="fixture",
        ),
        EvaluationCase(
            case_id="q2",
            split="dev",
            question="second",
            answerability="answerable",
            gold_chunk_ids=["b:0000"],
            reference_answer="second",
            notes="fixture",
        ),
    ]

    metrics = retrieval_metrics({"q1": ["a:0000"], "q2": ["a:0000"]}, cases, k=1)

    assert metrics["recall_at_1"] == 0.5
    assert metrics["mrr_at_1"] == 0.5
    assert metrics["ndcg_at_1"] == 0.5


def test_recall_at_k_measures_coverage_of_multiple_gold_chunks() -> None:
    cases = [
        EvaluationCase(
            case_id="q1",
            split="test",
            question="two supporting passages",
            answerability="answerable",
            gold_chunk_ids=["a:0000", "a:0007"],
            reference_answer="both passages independently state the fact",
            notes="fixture",
        )
    ]

    metrics = retrieval_metrics({"q1": ["a:0000", "b:0000"]}, cases, k=2)

    assert metrics["recall_at_2"] == 0.5
    assert metrics["mrr_at_2"] == 1.0
