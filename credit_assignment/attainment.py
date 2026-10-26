"""Persistent union hypervolume of coherent, identified research answers.

The archive remembers paid attainment; it never selects the current scientific
answer. Repeated material retains its first measurement instead of drawing
another opportunity for credit from evaluator variation.
"""

from functools import lru_cache
from fractions import Fraction
import math


def _frontier(points):
    retained = []
    # A later lexicographic point cannot dominate an earlier distinct point.
    for point in sorted(set(points), reverse=True):
        if not any(all(a <= b for a, b in zip(point, other)) for other in retained):
            retained.append(point)
    return tuple(sorted(retained))


@lru_cache(maxsize=131072)
def _volume(points):
    if not points:
        return 0.0
    if len(points) == 1:
        return math.prod(points[0])
    if len(points[0]) == 1:
        return max(p[0] for p in points)
    if len(points[0]) == 2:
        height, parts = 0.0, []
        for x, y in sorted(points, reverse=True):
            if y > height:
                parts.append(x*(y-height))
                height = y
        return math.fsum(parts)
    points = _frontier(points)
    if len(points) == 1:
        return math.prod(points[0])
    # The slab union is identical on every axis; fewer boundaries cost less.
    axis = min(range(len(points[0])), key=lambda j: len({p[j] for p in points}))
    parts, previous = [], 0.0
    for boundary in sorted({p[axis] for p in points}):
        projected = tuple(sorted({p[:axis]+p[axis+1:] for p in points if p[axis] >= boundary}))
        parts.append((boundary-previous)*_volume(projected))
        previous = boundary
    return math.fsum(parts)


def hypervolume(points):
    """Exact dominated union volume, referenced to zero; no coordinate maxima."""
    points = tuple(tuple(float(v) for v in p) for p in points)
    if points and (not points[0] or any(len(p) != len(points[0]) for p in points)):
        raise ValueError('Attainment vectors require a fixed nonempty dimension set')
    if any(not math.isfinite(v) or not 0 <= v <= 1 for p in points for v in p):
        raise ValueError('Attainment coordinates must be finite values in [0,1]')
    return _volume(tuple(sorted(set(points))))


class AttainmentArchive:
    def __init__(self, columns, records=None):
        self.columns = tuple(columns)
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError('Attainment columns must be distinct and nonempty')
        self.records = {}
        for identity, vector in (records or {}).items():
            vector=tuple(vector)
            if not isinstance(identity,str) or not identity or len(vector)!=len(self.columns):
                raise ValueError('An attainment point requires an identity and every column')
            hypervolume((vector,))
            self.records[identity]=vector

    @property
    def score(self):
        return hypervolume(self.records.values())

    def add(self, identity, vector):
        vector = tuple(vector)
        if not isinstance(identity, str) or not identity or len(vector) != len(self.columns):
            raise ValueError('An attainment point requires an identity and every column')
        hypervolume((vector,))
        if identity in self.records:
            return 0.0
        previous = self.score
        # All attempts still belong in the audit. This identity is credit memory,
        # not an admission rule for evidence, arguments, or model observations.
        self.records[identity] = vector
        return self.score-previous

    def as_record(self):
        return {'version': 1, 'columns': list(self.columns),
                'records': {k: list(v) for k, v in self.records.items()}}

    @classmethod
    def from_record(cls, value):
        if value.get('version') != 1:
            raise ValueError('Unknown attainment archive version')
        return cls(value['columns'], value['records'])


def portfolio_frontier(groups, capacity):
    """Exact additive portfolios, one coherent vector per independent group.

    Each group maps immutable material identity to its canonical vector. The
    cardinality dimension prevents pruning a smaller portfolio that can still
    use remaining slots. Witnesses preserve which whole answers made a sum.
    """
    if type(capacity) is not int or capacity < 1:
        raise ValueError('Portfolio capacity must be a positive integer')
    dimensions=next((len(v) for group in groups.values() for v in group.values()),0)
    if not dimensions:
        return [({},())], {'groups':len(groups),'largest_cardinality_frontier':1}
    # Sum the recorded decimal ratings exactly. Floating addition order must
    # not create artificial tradeoffs or extra Pareto points.
    scale=math.lcm(*(Fraction(str(v)).denominator
                     for group in groups.values() for vector in group.values() for v in vector))
    states={0:{(0,)*dimensions:{}}}
    largest=1
    for key,group in sorted(groups.items()):
        options={}
        for identity,vector in sorted(group.items()):
            hypervolume((vector,))
            if len(vector)!=dimensions:
                raise ValueError('Portfolio groups must use the same dimensions')
            exact=tuple(int(Fraction(str(v))*scale) for v in vector)
            options[exact]=identity
        frontier=set(_frontier(tuple(options)))
        options={v:identity for v,identity in options.items() if v in frontier}
        following={count:dict(rows) for count,rows in states.items()}
        for count,rows in states.items():
            if count>=capacity:
                continue
            target=following.setdefault(count+1,{})
            for accumulated,witness in rows.items():
                for vector,identity in options.items():
                    total=tuple(a+b for a,b in zip(accumulated,vector))
                    target.setdefault(total,{**witness,key:identity})
        states={count:{v:rows[v] for v in _frontier(tuple(rows))}
                for count,rows in following.items()}
        largest=max(largest,max(map(len,states.values())))
    # Dominance is safe across cardinalities only after all groups are consumed.
    complete={vector:witness for rows in states.values() for vector,witness in rows.items()}
    result=[(complete[v],tuple(coordinate/scale for coordinate in v))
            for v in _frontier(tuple(complete))]
    return result,{'groups':len(groups),'largest_cardinality_frontier':largest,
                   'final_frontier':len(result)}
