"""A short plain-English description of a record: what gets embedded as the record's text."""

from __future__ import annotations

from collections import Counter

from defects.labels import NAMES
from fusion.schema import HazardRecord

SEVERITY_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}


def describe(record: HazardRecord) -> str:
    """One sentence per fact: the light, the objects, the damage and the hazard score.

    The same words a person would type into a search box ("night", "pothole", "pedestrian"), so a
    text query lands near the records that match it.
    """
    parts = []
    if record.meta.time_of_day != "unknown":
        parts.append(f"A {record.meta.time_of_day}time road scene.")
    counts = Counter(o.label.replace("_", " ") for o in record.objects)
    if counts:
        listed = ", ".join(f"{n} {label}" for label, n in sorted(counts.items()))
        parts.append(f"Objects: {listed}.")
    if record.defects:
        names = sorted({NAMES.get(d.name, d.name) for d in record.defects})
        parts.append("Road damage: " + ", ".join(names) + ".")
    else:
        parts.append("No road damage found.")
    parts.append(f"Hazard score {round(100 * min(1.0, record.hazard_score))} out of 100.")
    return " ".join(parts)
