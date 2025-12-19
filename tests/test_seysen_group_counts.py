from dikin_utils import __reduction_ranges, group_non_overlapping


def test_group_non_overlapping_counts():
    sizes = range(60, 1200, 60)
    prev_count = 0
    for n in sizes:
        _, ranges = __reduction_ranges(n)
        levels = group_non_overlapping(ranges)
        count = len(levels)
        interval_lengths = sorted({k - i for i, _, k in ranges})
        print(
            f"n={n}: {count} non-overlapping groups; "
            f"{len(ranges)} ranges; lengths={interval_lengths}"
        )
        assert count > 0
        assert count >= prev_count
        prev_count = count
