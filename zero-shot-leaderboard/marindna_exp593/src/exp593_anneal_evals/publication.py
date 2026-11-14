"""Narrow replacement of owned score tables, preserving live human edits."""

from __future__ import annotations


def replace_tables(live: str, previous: str, replacements: dict[str, str]) -> str:
    marker = "<!-- exp593-anneal-final-results-v1 -->"
    if marker not in live or marker not in previous:
        raise ValueError("The single owned results comment cannot be identified")
    result = live
    for section, content in replacements.items():
        if section not in {"primary-results", "detailed-results"}:
            raise ValueError("Only generated score tables may be replaced")
        begin, end = f"<!-- {section}:start -->", f"<!-- {section}:end -->"
        for body in [live, previous]:
            if body.count(begin) != 1 or body.count(end) != 1:
                raise ValueError("A generated table marker is absent or duplicated")
        old = previous.split(begin, 1)[1].split(end, 1)[0]
        current = live.split(begin, 1)[1].split(end, 1)[0]
        if current != old:
            raise ValueError(
                "A human edited a generated table; review before replacing it"
            )
        prefix, rest = result.split(begin, 1)
        _, suffix = rest.split(end, 1)
        result = prefix + begin + "\n" + content.strip() + "\n" + end + suffix
    return result
