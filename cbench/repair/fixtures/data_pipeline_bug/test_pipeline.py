"""Tests for the data pipeline."""

from pipeline import (
    aggregate_by_category,
    deduplicate,
    filter_records,
    parse_csv,
    transform_pipeline,
)

SAMPLE_CSV = """id,name,value,category,tags
1,Alice,100.0,A,tag1,tag2
2,Bob,200.0,B,tag1
3,Carol,150.0,A,tag2,tag3
4,Dave,0.0,B,
5,Eve,300.0,A,tag1
"""

CSV_WITH_EMPTY_VALUE = """id,name,value,category,tags
1,Alice,100.0,A,tag1
2,Bob,,B,tag2
3,Carol,50.0,A,
"""

CSV_WITH_DUPLICATES = """id,name,value,category,tags
1,Alice,100.0,A,tag1
2,Bob,200.0,B,tag2
1,Alice_Updated,150.0,A,tag3
3,Carol,50.0,A,
"""


def test_parse_csv():
    records = parse_csv(SAMPLE_CSV)
    assert len(records) == 5
    assert records[0].name == "Alice"
    assert records[0].value == 100.0


def test_parse_csv_empty_value():
    """Empty value field should parse as 0.0, not crash."""
    records = parse_csv(CSV_WITH_EMPTY_VALUE)
    assert len(records) == 3
    assert records[1].value == 0.0


def test_filter_records_min_value():
    records = parse_csv(SAMPLE_CSV)
    filtered = filter_records(records, min_value=100.0)
    # Should include records with value >= 100.0 (Alice=100, Bob=200, Carol=150, Eve=300)
    assert len(filtered) == 4
    names = [r.name for r in filtered]
    assert "Alice" in names
    assert "Dave" not in names


def test_filter_records_by_category():
    records = parse_csv(SAMPLE_CSV)
    filtered = filter_records(records, category="A")
    assert all(r.category == "A" for r in filtered)


def test_filter_records_zero_min():
    """Records with value=0.0 should be included when min_value=0.0."""
    records = parse_csv(SAMPLE_CSV)
    filtered = filter_records(records, min_value=0.0)
    assert len(filtered) == 5  # All records including Dave with value=0.0


def test_aggregate_by_category():
    records = parse_csv(SAMPLE_CSV)
    agg = aggregate_by_category(records)
    assert "A" in agg
    assert "B" in agg
    assert agg["A"]["count"] == 3
    assert agg["A"]["total"] == 550.0
    assert abs(agg["A"]["average"] - 183.33) < 0.01


def test_deduplicate_keeps_last():
    """Dedup should keep the LAST occurrence of each ID."""
    records = parse_csv(CSV_WITH_DUPLICATES)
    deduped = deduplicate(records)
    assert len(deduped) == 3  # IDs: 1, 2, 3
    # ID 1 should be the updated version
    rec_1 = [r for r in deduped if r.id == 1][0]
    assert rec_1.name == "Alice_Updated"
    assert rec_1.value == 150.0


def test_full_pipeline():
    result = transform_pipeline(SAMPLE_CSV, min_value=100.0)
    assert "A" in result
    assert result["A"]["count"] >= 2  # Alice and Eve at minimum


def test_full_pipeline_with_empty_values():
    """Pipeline should handle CSV with empty values gracefully."""
    result = transform_pipeline(CSV_WITH_EMPTY_VALUE, min_value=0.0)
    assert "A" in result
    assert "B" in result
