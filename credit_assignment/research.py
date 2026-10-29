"""Episode credit and the copied nano-graphrag incidence yield controller."""

from dataclasses import dataclass
import json
import math

from .incidence_rarefaction import (
    ChannelSchema, ControllerConfig, EstimatorController,
    OBSERVATION_FAILED, OBSERVATION_OBSERVED,
    adaptive_marginal_credit_threshold,
)


@dataclass(frozen=True)
class ResearchStep:
    stop: bool
    score: float
    gain: float
    evaluated: bool
    reason: str
    requests_transition: bool = False


@dataclass(frozen=True)
class ResearchState:
    stop: bool
    score: float
    history: tuple[float, ...]
    failures: int
    reason: str
    rarefaction: dict | None


class ResearchController:
    """Deduplicate at this scope, never add up children's accumulated credit."""

    def __init__(self, columns, *, window=4, root_window=None, capacity=2, epoch="research-v2", initial=None, groups=None,
                 sum_qualities=False, point_scores=False, yield_threshold=None, scope_path=None):
        if yield_threshold is not None and (isinstance(yield_threshold, bool) or
                not isinstance(yield_threshold, (int, float)) or not math.isfinite(yield_threshold) or yield_threshold < 0):
            raise ValueError("Yield threshold must be finite and nonnegative")
        self.yield_threshold = yield_threshold
        self.columns = tuple(dict.fromkeys(columns))
        if not self.columns or window < 2 or capacity < 1:
            raise ValueError("Research control requires columns, window >= 2 and positive capacity")
        if root_window is not None and (type(root_window) is not int or root_window < 2):
            raise ValueError("Root yield window must be an integer >= 2")
        self.window, self.capacity, self.epoch = window, capacity, epoch
        self.root_window = window if root_window is None else root_window
        self.sum_qualities = sum_qualities
        self.point_scores = point_scores
        self.groups = groups or (self.columns,)
        if set(c for group in self.groups for c in group) != set(self.columns) or any(not g for g in self.groups):
            raise ValueError("Credit groups must cover the declared columns")
        self.seen = {column: set() for column in self.columns}
        self.qualities = {column: {} for column in self.columns}
        self.quality_resolution = None
        if initial is not None:
            for column, key in initial.identities:
                if column in self.seen:
                    self.seen[column].add(key)
            self._qualities(initial)
        self.history = [self._score()]
        self.window_start = 0
        self.failures = 0
        self.stop = False
        self.reason = "continue"
        self.rarefaction = None
        self._incidence = EstimatorController(
            scope_path=tuple(scope_path or (("research", epoch),)), epoch=epoch,
            channel_schema=ChannelSchema.single("scientific_credit"),
            control=ControllerConfig.uniform(("scientific_credit",),
                gamma=yield_threshold or 0.0, rho=0.0, streak_length=1),
            window_size=(self.root_window if scope_path and len(scope_path) == 1 else window),
            subsample_size=max(1, (self.root_window if scope_path and len(scope_path) == 1 else window) // 2),
            threshold_adapter=(adaptive_marginal_credit_threshold
                if yield_threshold is None else None),
        )
        self._observed_seen = {column: set(keys) for column, keys in self.seen.items()}
        self._observed_qualities = {column: dict(rows) for column, rows in self.qualities.items()}

    def _earned_identities(self):
        """Identity atoms earned since this episode's last completed own unit."""
        earned = [json.dumps(["identity", column, key], separators=(",", ":"))
            for column, keys in self.seen.items()
            for key in keys - self._observed_seen[column]]
        earned.extend(json.dumps(["quality", column, key, quality], separators=(",", ":"))
            for column, rows in self.qualities.items()
            for key, quality in rows.items()
            if quality > self._observed_qualities[column].get(key, 0.0))
        return sorted(earned)

    def _observe_credit_snapshot(self):
        self._observed_seen = {column: set(keys) for column, keys in self.seen.items()}
        self._observed_qualities = {column: dict(rows) for column, rows in self.qualities.items()}

    def _qualities(self, value):
        if self.sum_qualities or self.point_scores:
            # Scientific projection is a complete per-subject snapshot, not an event counter.
            self.qualities = {column: {} for column in self.columns}
        if value.quality_resolution is not None:
            resolution = value.quality_resolution
            if isinstance(resolution, bool) or not isinstance(resolution, (int, float)) or not math.isfinite(resolution) or not 0 < resolution <= 1:
                raise ValueError("Quality resolution must be finite and between zero (exclusive) and one")
            self.quality_resolution = resolution
        for column, key, quality in value.qualities:
            if column not in self.qualities:
                raise ValueError("Credit column is outside this episode's goal")
            if (isinstance(quality, bool) or not isinstance(quality, (int, float))
                    or not math.isfinite(quality) or quality < 0
                    or (not (self.point_scores or self.sum_qualities) and quality > 1)):
                raise ValueError("Credit must be finite and nonnegative; bounded qualities cannot exceed one")
            self.qualities[column][key] = quality

    def _score(self):
        if self.point_scores:
            return round(sum(sum(values.values()) for values in self.qualities.values()), 10)
        if self.sum_qualities:
            return self._weighted({c: sum(self.qualities[c].values()) for c in self.columns})
        values = {c: max(min(self.capacity, len(self.seen[c])) / self.capacity,
                         max(self.qualities[c].values(), default=0)) for c in self.columns}
        return self._weighted(values)

    def _weighted(self, values):
        return sum(sum(values[c] for c in group) / len(group) for group in self.groups) / len(self.groups)

    def _maximum(self):
        if self.point_scores or self.sum_qualities:
            return None
        return 1.0

    def observe(self, unit_label, value, *, is_root):
        from ..case_reports.research.records import Credit
        if not isinstance(value, Credit):
            raise TypeError("ResearchController requires Credit")
        previous = self.history[-1]
        score = self._score()
        if not value.evaluated:
            self.failures += 1
        else:
            for column, key in value.identities:
                if column not in self.seen:
                    raise ValueError("Credit column is outside this episode's goal")
                self.seen[column].add(key)
            self._qualities(value)
            score = self._score()
        # Failed assessment preserves credit but still contributes zero realized yield.
        self.history.append(score)
        if score < previous:
            # A changed scientific sheet begins a new estimation epoch while
            # the score history remains available for audit.
            self.window_start = len(self.history) - 1
            self._incidence = self._incidence.transitioned(
                f"{self.epoch}:revision:{len(self.history)}")
            self._observe_credit_snapshot()
            earned = []
        else:
            earned = self._earned_identities() if value.evaluated else []
            if value.evaluated:
                self._observe_credit_snapshot()
        incidence = self._incidence.advance(
            unit_label, earned,
            observation_status=OBSERVATION_OBSERVED if value.evaluated else OBSERVATION_FAILED,
            is_root=is_root,
        )
        complete = self._maximum() is not None and score >= self._maximum()
        self.stop = complete or (incidence.stop and self.yield_threshold != 0)
        self.reason = "complete" if complete else (
            incidence.verdict.outcome if value.evaluated or self.stop else "failed_observation")
        self.rarefaction = {"should_yield": self.stop, "outcome": self.reason,
            "incidence": incidence.as_record()}
        return ResearchStep(self.stop, score, score - previous, value.evaluated, self.reason)

    def state(self):
        return ResearchState(self.stop, self._score(), tuple(self.history),
                             self.failures, self.reason, self.rarefaction)

    def refresh(self, value):
        """Refresh shared credit without inventing a completed unit observation.

        A suspended parent's current score follows child work immediately. Its
        observation history still measures net yield between its own completed
        units, so a child return cannot lose its gain merely because it was
        already visible. The next observe handles any net retraction normally.
        """
        from ..case_reports.research.records import Credit
        if not self.point_scores or not isinstance(value, Credit):
            raise TypeError("Shared refresh requires point-score Credit")
        if not value.evaluated:
            return 0.0
        previous = self._score()
        self._qualities(value)
        return self._score() - previous

    def transitioned(self, epoch):
        return ResearchController(self.columns, window=self.window, root_window=self.root_window, capacity=self.capacity, epoch=epoch,
                                  groups=self.groups, sum_qualities=self.sum_qualities, point_scores=self.point_scores,
                                  yield_threshold=self.yield_threshold, scope_path=self._incidence.scope_path)
