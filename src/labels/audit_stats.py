"""Interval arithmetic for the report audit's random controls.

Controls are a uniform draw of about 50 pairs from roughly 2,500 per finding,
so an unweighted binomial interval fits them and the finite-population
correction (under 2%) is ignored. Enriched strata never enter these
estimates.
"""

import math

Z95 = 1.959963984540054


def wilson(k, n, z=Z95):
    """95% Wilson score interval for k successes in n draws."""
    if n <= 0:
        raise ValueError("an interval needs at least one draw")
    if not 0 <= k <= n:
        raise ValueError("k must lie in [0, n]")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def trigger(k, n, width=0.20):
    """Decision D4: the extra control batch is due when the interval is wider than `width`."""
    lo, hi = wilson(k, n)
    return {"errors": int(k), "controls": int(n), "rate": k / n, "lower": lo, "upper": hi,
            "width": hi - lo, "fires": hi - lo > width}


def cohen_kappa(first, second):
    """Agreement between two passes over the same items, corrected for chance.

    Returns None when chance agreement is total (every answer identical in
    both passes), where kappa is undefined; raw agreement then says it all.
    """
    first, second = list(first), list(second)
    if len(first) != len(second) or not first:
        raise ValueError("kappa needs two equal, non-empty sequences")
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    labels = set(first) | set(second)
    expected = sum((first.count(v) / n) * (second.count(v) / n) for v in labels)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def go_no_go(per_finding, threshold=5, deadline_passed=False):
    """Decision D5 over {finding: {"decidable": bool, "agreed_target_errors": int}}.

    Go as soon as any finding reaches the threshold; no-go only when every
    finding is decidable and none does; otherwise undecided, which becomes
    audit-only once the deadline has passed.
    """
    reached = sorted(f for f, s in per_finding.items() if s["agreed_target_errors"] >= threshold)
    if reached:
        return {"decision": "go", "findings": reached}
    if all(s["decidable"] for s in per_finding.values()):
        return {"decision": "no-go", "findings": []}
    pending = sorted(f for f, s in per_finding.items() if not s["decidable"])
    return {"decision": "audit-only" if deadline_passed else "undecided", "findings": pending}
