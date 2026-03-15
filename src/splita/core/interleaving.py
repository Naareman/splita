"""Interleaving experiments for ranking/search comparison.

Team Draft and Balanced interleaving methods. Measures preference
by crediting clicks to the team (A or B) whose item was clicked.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest  # type: ignore[attr-defined]

from splita._types import InterleavingResult
from splita._validation import check_in_range, check_one_of, format_error

ArrayLike = list | tuple | np.ndarray


def _team_draft_interleave(
    ranking_a: list, ranking_b: list, *, rng: np.random.Generator
) -> tuple[list, list[int]]:
    """Team Draft interleaving: alternately pick items from A and B.

    Returns the interleaved list and a team assignment array where
    0 = team A and 1 = team B.
    """
    interleaved: list = []
    teams: list[int] = []
    seen: set = set()
    idx_a, idx_b = 0, 0

    # Randomly choose who picks first
    a_first = bool(rng.integers(2) == 0)

    while True:
        prev_len = len(interleaved)

        order = [("a", 0), ("b", 1)] if a_first else [("b", 1), ("a", 0)]

        for label, team_id in order:
            idx = idx_a if label == "a" else idx_b
            source = ranking_a if label == "a" else ranking_b
            while idx < len(source) and source[idx] in seen:
                idx += 1
            if idx < len(source):
                item = source[idx]
                interleaved.append(item)
                teams.append(team_id)
                seen.add(item)
                idx += 1
                if label == "a":
                    idx_a = idx
                else:
                    idx_b = idx

        # Stop if no new items were added or both lists exhausted
        if len(interleaved) == prev_len:
            break
        if idx_a >= len(ranking_a) and idx_b >= len(ranking_b):
            break

    return interleaved, teams


def _balanced_interleave(ranking_a: list, ranking_b: list) -> tuple[list, list[int]]:
    """Balanced interleaving: merge by rank, credit both if tied.

    Items at the same rank get priority based on their rank in each list.
    """
    interleaved: list = []
    teams: list[int] = []
    seen: set = set()

    rank_a = {item: i for i, item in enumerate(ranking_a)}
    rank_b = {item: i for i, item in enumerate(ranking_b)}
    all_items = list(dict.fromkeys(ranking_a + ranking_b))

    # Sort by minimum rank in either list
    def sort_key(item: object) -> float:
        ra = rank_a.get(item, float("inf"))
        rb = rank_b.get(item, float("inf"))
        return min(ra, rb)

    all_items.sort(key=sort_key)

    for item in all_items:
        if item in seen:
            continue
        seen.add(item)
        interleaved.append(item)
        ra = rank_a.get(item, float("inf"))
        rb = rank_b.get(item, float("inf"))
        if ra <= rb:
            teams.append(0)  # team A
        else:
            teams.append(1)  # team B

    return interleaved, teams


class InterleavingExperiment:
    """Interleaving experiment for comparing two ranking systems.

    Interleaves ranked lists from systems A and B, then measures
    user preference through click attribution.

    Parameters
    ----------
    method : ``"team_draft"`` or ``"balanced"``, default ``"team_draft"``
        Interleaving method.
    alpha : float, default 0.05
        Significance level for the preference test.

    Examples
    --------
    >>> rankings_a = [[1, 2, 3], [1, 3, 2]]
    >>> rankings_b = [[3, 2, 1], [2, 1, 3]]
    >>> clicks = [[0], [1]]  # clicked positions
    >>> exp = InterleavingExperiment(alpha=0.05)
    >>> r = exp.run(rankings_a, rankings_b, clicks)
    >>> r.n_queries
    2
    """

    def __init__(
        self,
        *,
        method: str = "team_draft",
        alpha: float = 0.05,
    ) -> None:
        check_one_of(method, "method", ["team_draft", "balanced"])
        check_in_range(
            alpha,
            "alpha",
            0.0,
            1.0,
            hint="typical values are 0.05, 0.01, or 0.10",
        )
        self._method = method
        self._alpha = alpha

    def run(
        self,
        rankings_a: list[list],
        rankings_b: list[list],
        clicks: list[list[int]],
    ) -> InterleavingResult:
        """Run the interleaving experiment.

        Parameters
        ----------
        rankings_a : list of list
            Ranked lists from system A, one per query.
        rankings_b : list of list
            Ranked lists from system B, one per query.
        clicks : list of list of int
            Clicked positions (0-indexed) in the interleaved list,
            one list per query.

        Returns
        -------
        InterleavingResult
            Preference statistics and significance.

        Raises
        ------
        ValueError
            If inputs are empty or have mismatched lengths.
        """
        if not isinstance(rankings_a, list) or len(rankings_a) == 0:
            raise ValueError(
                format_error(
                    "`rankings_a` can't be empty.",
                    "received an empty or non-list input.",
                    hint="pass a list of ranked lists, one per query.",
                )
            )
        if not isinstance(rankings_b, list) or len(rankings_b) == 0:
            raise ValueError(
                format_error(
                    "`rankings_b` can't be empty.",
                    "received an empty or non-list input.",
                    hint="pass a list of ranked lists, one per query.",
                )
            )
        if not isinstance(clicks, list) or len(clicks) == 0:
            raise ValueError(
                format_error(
                    "`clicks` can't be empty.",
                    "received an empty or non-list input.",
                    hint="pass a list of click position lists, one per query.",
                )
            )
        if len(rankings_a) != len(rankings_b):
            raise ValueError(
                format_error(
                    "`rankings_a` and `rankings_b` must have the same length.",
                    f"rankings_a has {len(rankings_a)} queries, "
                    f"rankings_b has {len(rankings_b)} queries.",
                )
            )
        if len(rankings_a) != len(clicks):
            raise ValueError(
                format_error(
                    "`rankings_a` and `clicks` must have the same length.",
                    f"rankings_a has {len(rankings_a)} queries, clicks has {len(clicks)} queries.",
                )
            )

        rng = np.random.default_rng(42)
        wins_a = 0
        wins_b = 0

        for ra, rb, click_positions in zip(rankings_a, rankings_b, clicks, strict=False):
            if self._method == "team_draft":
                _interleaved2, teams = _team_draft_interleave(ra, rb, rng=rng)
            else:
                _interleaved, teams = _balanced_interleave(ra, rb)

            # Credit clicks to teams
            clicks_a = 0
            clicks_b = 0
            for pos in click_positions:
                if 0 <= pos < len(teams):
                    if teams[pos] == 0:
                        clicks_a += 1
                    else:
                        clicks_b += 1

            if clicks_a > clicks_b:
                wins_a += 1
            elif clicks_b > clicks_a:
                wins_b += 1
            # ties don't count

        n_decided = wins_a + wins_b
        n_queries = len(rankings_a)

        if n_decided == 0:
            pvalue = 1.0
        else:
            # Binomial test: H0: p = 0.5
            result = binomtest(wins_a, n_decided, 0.5, alternative="two-sided")
            pvalue = float(result.pvalue)

        preference_a = wins_a / n_queries if n_queries > 0 else 0.0
        preference_b = wins_b / n_queries if n_queries > 0 else 0.0
        delta = preference_a - preference_b
        significant = pvalue < self._alpha

        winner = ("A" if wins_a > wins_b else "B") if significant else "tie"

        return InterleavingResult(
            preference_a=preference_a,
            preference_b=preference_b,
            pvalue=pvalue,
            significant=significant,
            winner=winner,
            n_queries=n_queries,
            delta=delta,
        )
