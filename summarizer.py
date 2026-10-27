"""TextRank-style extractive summarization and lightweight ROUGE evaluation."""

import re
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LENGTH_RATIOS = {"Short": 0.20, "Medium": 0.35, "Long": 0.50}


def split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", cleaned)
    return [part.strip() for part in parts if part.strip()]


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _pagerank(similarity_matrix: np.ndarray, damping: float = 0.85,
              max_iter: int = 100, tolerance: float = 1e-6) -> np.ndarray:
    n = similarity_matrix.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    matrix = similarity_matrix.astype(float).copy()
    np.fill_diagonal(matrix, 0.0)
    row_sums = matrix.sum(axis=1)
    transition = np.zeros_like(matrix)
    for i in range(n):
        transition[i] = matrix[i] / row_sums[i] if row_sums[i] > 0 else np.full(n, 1.0 / n)
    scores = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * transition.T.dot(scores)
        if np.linalg.norm(new_scores - scores, ord=1) < tolerance:
            scores = new_scores
            break
        scores = new_scores
    total = scores.sum()
    return scores / total if total else np.full(n, 1.0 / n)


def summarize_text(text: str, length: str = "Medium") -> Tuple[str, List[Dict]]:
    sentences = split_sentences(text)
    if not sentences:
        return "", []
    if len(sentences) <= 2:
        return " ".join(sentences), [
            {"index": i + 1, "score": 1.0 / len(sentences), "sentence": sentence}
            for i, sentence in enumerate(sentences)
        ]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        vectors = vectorizer.fit_transform(sentences)
        similarity = cosine_similarity(vectors)
    except ValueError:
        return sentences[0], [
            {"index": i + 1, "score": 0.0, "sentence": sentence}
            for i, sentence in enumerate(sentences)
        ]
    scores = _pagerank(similarity)
    ratio = LENGTH_RATIOS.get(length, LENGTH_RATIOS["Medium"])
    target_count = max(1, min(len(sentences), int(np.ceil(len(sentences) * ratio))))
    ranked_indices = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    selected = sorted(ranked_indices[:target_count])
    summary = " ".join(sentences[i] for i in selected)
    ranking = [
        {"index": i + 1, "score": float(scores[i]), "sentence": sentences[i]}
        for i in ranked_indices
    ]
    return summary, ranking


def _tokens(text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _f1(overlap: int, candidate_count: int, reference_count: int) -> float:
    if candidate_count == 0 or reference_count == 0 or overlap == 0:
        return 0.0
    precision = overlap / candidate_count
    recall = overlap / reference_count
    return 2 * precision * recall / (precision + recall)


def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1)))


def _lcs_length(a: List[str], b: List[str]) -> int:
    # Dynamic programming; adequate for typical student project text sizes.
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def evaluate_against_reference(candidate: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2 and ROUGE-L F1 scores (single reference)."""
    cand = _tokens(candidate)
    ref = _tokens(reference)
    results = {}
    for n, name in [(1, "rouge1_f1"), (2, "rouge2_f1")]:
        cand_counts = _ngram_counts(cand, n)
        ref_counts = _ngram_counts(ref, n)
        overlap = sum((cand_counts & ref_counts).values())
        results[name] = _f1(overlap, sum(cand_counts.values()), sum(ref_counts.values()))
    lcs = _lcs_length(cand, ref)
    results["rougeL_f1"] = _f1(lcs, len(cand), len(ref))
    return results
