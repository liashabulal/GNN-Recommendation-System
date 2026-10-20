"""Metric tests on small impressions whose values can be checked by hand."""

import math

import pytest

from recsys.evaluation.metrics import auc, mrr, ndcg


def test_perfect_ranking():
    labels, scores = [1, 0, 0], [0.9, 0.5, 0.1]
    assert auc(labels, scores) == 1.0
    assert mrr(labels, scores) == 1.0
    assert ndcg(labels, scores, k=5) == 1.0


def test_click_ranked_last():
    # The only click is at rank 3 of 3.
    labels, scores = [1, 0, 0], [0.1, 0.9, 0.5]
    assert auc(labels, scores) == 0.0
    assert mrr(labels, scores) == pytest.approx(1 / 3)
    assert ndcg(labels, scores, k=5) == pytest.approx(1 / math.log2(4))  # = 0.5


def test_two_clicks():
    # Ranked order: click, non-click, click, non-click (clicks at ranks 1 and 3).
    labels, scores = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]
    # Pairs (click beats non-click): 0.9 beats both, 0.7 beats only 0.1 -> 3 of 4.
    assert auc(labels, scores) == pytest.approx(3 / 4)
    # MIND's MRR averages 1/rank over all clicks: (1/1 + 1/3) / 2.
    assert mrr(labels, scores) == pytest.approx(2 / 3)
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg(labels, scores, k=5) == pytest.approx(dcg / ideal)


def test_ndcg_cutoff():
    # One click at rank 6: outside the top 5, inside the top 10.
    labels = [0, 0, 0, 0, 0, 1, 0]
    scores = [7, 6, 5, 4, 3, 2, 1]
    assert ndcg(labels, scores, k=5) == 0.0
    assert ndcg(labels, scores, k=10) == pytest.approx(1 / math.log2(7))


def test_ties_use_expected_value():
    # Click tied with a non-click: it lands at rank 1 or 2 with equal probability.
    labels, scores = [1, 0], [0.5, 0.5]
    assert auc(labels, scores) == 0.5
    assert mrr(labels, scores) == pytest.approx(0.5 * 1 + 0.5 * 1 / 2)
    assert ndcg(labels, scores, k=5) == pytest.approx(0.5 * 1 + 0.5 / math.log2(3))


def test_ties_do_not_depend_on_candidate_order():
    a = ([1, 0, 0, 0], [1.0, 0.0, 0.0, 0.0])
    b = ([0, 0, 0, 1], [0.0, 0.0, 0.0, 0.0])
    c = ([1, 0, 0, 0], [0.0, 0.0, 0.0, 0.0])
    # b and c are the same impression with candidates in a different order.
    for metric in (auc, mrr, lambda y, s: ndcg(y, s, k=2)):
        assert metric(*b) == pytest.approx(metric(*c))
    assert mrr(*a) == 1.0


def test_tie_group_below_a_clear_winner():
    # One non-click scores highest; the click is tied with two non-clicks at ranks 2-4.
    labels, scores = [0, 1, 0, 0], [0.9, 0.2, 0.2, 0.2]
    assert mrr(labels, scores) == pytest.approx((1 / 2 + 1 / 3 + 1 / 4) / 3)
    # AUC: the click beats 0 of 1 higher non-click and ties the other 2 -> (0 + 0.5 + 0.5) / 3.
    assert auc(labels, scores) == pytest.approx(1 / 3)


def test_auc_undefined_without_non_clicks():
    assert math.isnan(auc([1, 1], [0.3, 0.7]))


def test_no_click_raises():
    with pytest.raises(ValueError):
        mrr([0, 0], [0.1, 0.2])


def test_non_binary_labels_raise():
    with pytest.raises(ValueError):
        ndcg([2, 0], [0.1, 0.2], k=5)
