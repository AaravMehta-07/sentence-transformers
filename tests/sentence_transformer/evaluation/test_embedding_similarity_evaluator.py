"""
Tests the correct computation of evaluation scores from EmbeddingSimilarityEvaluator
"""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

SENTENCES1 = ["A person on a horse jumps over a broken down airplane.", "Children smiling and waving at camera"]
SENTENCES2 = ["A person is outdoors, on a horse.", "A person is at a diner, ordering an omelette."]
SCORES = [0.9, 0.1]
MODEL_DEFAULT_SENTENCES1 = [
    "hello world",
    "sentence transformers",
    "embedding model",
    "text vector",
    "hello sentence",
    "world model",
    "transformers vector",
    "embedding text",
    "hello vector",
    "world sentence",
    "model text",
    "transformers embedding",
]
MODEL_DEFAULT_SENTENCES2 = [
    "sentence model",
    "hello vector",
    "world text",
    "transformers embedding",
    "vector model",
    "sentence text",
    "hello embedding",
    "world transformers",
    "model world",
    "text sentence",
    "vector embedding",
    "hello model",
]
MODEL_DEFAULT_SCORES = [0.9, 0.1, 0.4, 0.7, 0.3, 0.8, 0.2, 0.6, 0.5, 0.1, 0.9, 0.2]


@pytest.mark.parametrize("precision", [None, "float32", "int8", "uint8", "binary", "ubinary"])
def test_embedding_similarity_evaluator_precisions(
    stsb_bert_tiny_model: SentenceTransformer, precision: str | None
) -> None:
    """Every supported precision must produce metrics."""
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=SENTENCES1,
        sentences2=SENTENCES2,
        scores=SCORES,
        precision=precision,
        name="stsb_dev",
    )
    metrics = evaluator(stsb_bert_tiny_model)

    assert evaluator.primary_metric in metrics
    assert np.isfinite(metrics[evaluator.primary_metric])


def test_embedding_similarity_evaluator_follows_each_model_default(
    word_embeddings_model: SentenceTransformer, tmp_path: Path
) -> None:
    cosine_model = word_embeddings_model
    dot_model = copy.deepcopy(word_embeddings_model)
    cosine_model.similarity_fn_name = "cosine"
    dot_model.similarity_fn_name = "dot"

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=MODEL_DEFAULT_SENTENCES1,
        sentences2=MODEL_DEFAULT_SENTENCES2,
        scores=MODEL_DEFAULT_SCORES,
    )
    evaluator.csv_file = "checkpoint_results.csv"
    cosine_metrics = evaluator(cosine_model, output_path=str(tmp_path), epoch=0, steps=0)
    assert evaluator.primary_metric == "spearman_cosine"
    assert evaluator.primary_metric in cosine_metrics

    dot_metrics = evaluator(dot_model, output_path=str(tmp_path), epoch=1, steps=10)
    fresh_dot_metrics = EmbeddingSimilarityEvaluator(
        sentences1=MODEL_DEFAULT_SENTENCES1,
        sentences2=MODEL_DEFAULT_SENTENCES2,
        scores=MODEL_DEFAULT_SCORES,
        write_csv=False,
    )(dot_model)

    assert dot_metrics == fresh_dot_metrics
    assert evaluator.primary_metric == "spearman_dot"
    assert evaluator.primary_metric in dot_metrics
    cosine_again_metrics = evaluator(cosine_model, output_path=str(tmp_path), epoch=2, steps=20)
    assert evaluator.primary_metric == "spearman_cosine"
    assert evaluator.primary_metric in cosine_again_metrics
    assert evaluator.csv_file == "checkpoint_results.csv"

    with (tmp_path / "checkpoint_results.csv").open(newline="", encoding="utf-8") as file:
        cosine_csv = list(csv.reader(file))
    with (tmp_path / "checkpoint_dot_results.csv").open(newline="", encoding="utf-8") as file:
        dot_csv = list(csv.reader(file))

    assert cosine_csv[0] == ["epoch", "steps", "cosine_pearson", "cosine_spearman"]
    assert dot_csv[0] == ["epoch", "steps", "dot_pearson", "dot_spearman"]
    assert len(cosine_csv) == 3
    assert len(cosine_csv[1]) == len(cosine_csv[0])
    assert len(dot_csv[1]) == len(dot_csv[0])
    assert [float(value) for value in cosine_csv[1]] == pytest.approx(
        [0, 0, cosine_metrics["pearson_cosine"], cosine_metrics["spearman_cosine"]]
    )
    assert [float(value) for value in cosine_csv[2]] == pytest.approx(
        [2, 20, cosine_again_metrics["pearson_cosine"], cosine_again_metrics["spearman_cosine"]]
    )
    assert [float(value) for value in dot_csv[1]] == pytest.approx(
        [1, 10, dot_metrics["pearson_dot"], dot_metrics["spearman_dot"]]
    )


def test_embedding_similarity_evaluator_keeps_explicit_similarity_functions(
    word_embeddings_model: SentenceTransformer,
) -> None:
    cosine_model = word_embeddings_model
    dot_model = copy.deepcopy(word_embeddings_model)
    cosine_model.similarity_fn_name = "cosine"
    dot_model.similarity_fn_name = "dot"

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=MODEL_DEFAULT_SENTENCES1,
        sentences2=MODEL_DEFAULT_SENTENCES2,
        scores=MODEL_DEFAULT_SCORES,
        similarity_fn_names=["cosine"],
        write_csv=False,
    )
    evaluator(cosine_model)
    metrics = evaluator(dot_model)

    assert evaluator.similarity_fn_names == ["cosine"]
    assert set(metrics) == {"pearson_cosine", "spearman_cosine"}
    assert evaluator.primary_metric == "spearman_cosine"
    assert evaluator.primary_metric in metrics


def test_embedding_similarity_evaluator_does_not_read_csv_when_disabled(
    word_embeddings_model: SentenceTransformer, tmp_path: Path
) -> None:
    word_embeddings_model.similarity_fn_name = "dot"
    csv_path = tmp_path / "similarity_evaluation_results.csv"
    csv_path.write_bytes(b"\xff\xfe\xfd")

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=MODEL_DEFAULT_SENTENCES1,
        sentences2=MODEL_DEFAULT_SENTENCES2,
        scores=MODEL_DEFAULT_SCORES,
        write_csv=False,
    )
    metrics = evaluator(word_embeddings_model, output_path=str(tmp_path))

    assert evaluator.primary_metric == "spearman_dot"
    assert evaluator.primary_metric in metrics
    assert csv_path.read_bytes() == b"\xff\xfe\xfd"


def test_embedding_similarity_evaluator_main_similarity_overrides_model_default(
    word_embeddings_model: SentenceTransformer,
) -> None:
    word_embeddings_model.similarity_fn_name = "dot"
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=MODEL_DEFAULT_SENTENCES1,
        sentences2=MODEL_DEFAULT_SENTENCES2,
        scores=MODEL_DEFAULT_SCORES,
        main_similarity="cosine",
        write_csv=False,
    )
    metrics = evaluator(word_embeddings_model)

    assert evaluator.similarity_fn_names == ["cosine"]
    assert set(metrics) == {"pearson_cosine", "spearman_cosine"}
    assert evaluator.primary_metric == "spearman_cosine"
    assert evaluator.primary_metric in metrics


def test_embedding_similarity_evaluator_rejects_conflicting_csv_headers(
    word_embeddings_model: SentenceTransformer, tmp_path: Path
) -> None:
    word_embeddings_model.similarity_fn_name = "dot"
    base_csv_path = tmp_path / "similarity_evaluation_results.csv"
    dot_csv_path = tmp_path / "similarity_evaluation_dot_results.csv"
    base_csv_path.write_text("epoch,steps,stale\n", encoding="utf-8")
    dot_csv_path.write_text("epoch,steps,also_stale\n", encoding="utf-8")

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=MODEL_DEFAULT_SENTENCES1,
        sentences2=MODEL_DEFAULT_SENTENCES2,
        scores=MODEL_DEFAULT_SCORES,
    )
    with pytest.raises(ValueError, match="expected CSV headers"):
        evaluator(word_embeddings_model, output_path=str(tmp_path))
