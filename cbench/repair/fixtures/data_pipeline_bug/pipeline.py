"""Data pipeline that processes records with multiple bugs."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field


@dataclass
class Record:
    id: int
    name: str
    value: float
    category: str
    tags: list[str] = field(default_factory=list)


def parse_csv(csv_text: str) -> list[Record]:
    """Parse CSV text into records.

    BUG: Doesn't handle empty value field (crashes with ValueError on float(''))
    """
    records = []
    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        record = Record(
            id=int(row["id"]),
            name=row["name"],
            value=float(row["value"]),  # BUG: crashes on empty string
            category=row["category"],
            tags=row.get("tags", "").split(",") if row.get("tags") else [],
        )
        records.append(record)
    return records


def filter_records(records: list[Record], min_value: float = 0.0, category: str | None = None) -> list[Record]:
    """Filter records by minimum value and/or category.

    BUG: Uses > instead of >= for min_value comparison
    """
    result = []
    for r in records:
        if r.value > min_value:  # BUG: should be >= min_value
            if category is None or r.category == category:
                result.append(r)
    return result


def aggregate_by_category(records: list[Record]) -> dict[str, dict]:
    """Aggregate records by category.

    BUG: count is initialized to 0 but should be 1 for the first record,
    and total doesn't include the first record's value.
    """
    agg: dict[str, dict] = {}
    for r in records:
        if r.category not in agg:
            # BUG: count starts at 0, total starts at 0, then the loop body
            # adds count += 1 and total += value, so first record works BUT...
            agg[r.category] = {"count": 0, "total": 0.0, "names": []}
        agg[r.category]["count"] += 1
        agg[r.category]["total"] += r.value
        agg[r.category]["names"].append(r.name)

    # Compute averages
    for cat_data in agg.values():
        # BUG: division by count, but if count got corrupted above this crashes
        cat_data["average"] = cat_data["total"] / cat_data["count"]

    return agg


def deduplicate(records: list[Record]) -> list[Record]:
    """Remove duplicate records by ID, keeping the last occurrence.

    BUG: Keeps the first occurrence instead of the last.
    """
    seen: dict[int, Record] = {}
    for r in records:
        if r.id not in seen:  # BUG: should always overwrite to keep last
            seen[r.id] = r
    return list(seen.values())


def transform_pipeline(csv_text: str, min_value: float = 0.0, category: str | None = None) -> dict:
    """Full pipeline: parse -> deduplicate -> filter -> aggregate."""
    records = parse_csv(csv_text)
    records = deduplicate(records)
    records = filter_records(records, min_value=min_value, category=category)
    return aggregate_by_category(records)
