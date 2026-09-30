"""Candidate identity and optional video time bounds for PAS reranking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Interval:
    video: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def overlap(self, other: "Interval") -> float:
        if self.video != other.video:
            return 0.0
        return max(0.0, min(self.end, other.end) - max(self.start, other.start))


@dataclass(frozen=True)
class Segment:
    segment_id: int
    video: str
    start: float | None
    end: float | None

    @property
    def has_time_bounds(self) -> bool:
        return self.start is not None and self.end is not None

    @property
    def interval(self) -> Interval:
        if self.start is None or self.end is None:
            raise ValueError("Segment has no time bounds")
        return Interval(self.video, self.start, self.end)
