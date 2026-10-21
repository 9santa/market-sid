"""Shared Amazon metadata text and interaction filtering helpers."""

import html
import json
import re

import polars as pl


WHITESPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"<[^>]+>")

IGNORE_DETAIL_KEYS = {
    "ASIN",
    "UPC",
    "EAN",
    "ISBN",
    "Best Sellers Rank",
    "Customer Reviews",
    "Date First Available",
}


def clean(value) -> str:
    if value is None:
        return ""

    value = str(value)
    value = html.unescape(value)
    value = HTML_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value)

    return value.strip()


def clean_list(values, limit=None) -> list[str]:
    if not values:
        return []

    result = [clean(v) for v in values if clean(v)]

    if limit is not None:
        result = result[:limit]

    return result


def format_details(raw_details) -> str:
    if not raw_details:
        return ""

    try:
        if isinstance(raw_details, str):
            details = json.loads(raw_details)
        else:
            details = raw_details
    except (json.JSONDecodeError, TypeError):
        return ""

    if not isinstance(details, dict):
        return ""

    parts = []

    for key, value in details.items():
        if key in IGNORE_DETAIL_KEYS:
            continue

        key = clean(key)
        value = clean(value)

        if not key or not value:
            continue

        # Guard against mega-long fields
        value = value[:200]

        parts.append(f"{key}: {value}")

        if len(parts) >= 12:
            break

    return "; ".join(parts)


def build_item_text(row: dict) -> str:
    parts = []

    title = clean(row.get("title"))
    if title:
        parts.append(f"Title: {title}")

    store = clean(row.get("store"))
    if store:
        parts.append(f"Brand or store: {store}")

    categories = clean_list(
        row.get("categories"),
        limit=8,
    )
    if categories:
        parts.append("Categories: " + " > ".join(categories))

    features = clean_list(
        row.get("features"),
        limit=10,
    )
    if features:
        parts.append("Features: " + "; ".join(features))

    descriptions = clean_list(
        row.get("description"),
        limit=3,
    )
    if descriptions:
        parts.append("Description: " + " ".join(descriptions))

    details = format_details(row.get("details"))
    if details:
        parts.append("Details: " + details)

    text = "\n".join(parts)

    return text[:4000]



def iterative_k_core(
    df: pl.DataFrame,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pl.DataFrame:
    iteration = 0

    while True:
        iteration += 1
        old_rows = len(df)

        valid_users = (
            df.group_by("user_id")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_user_interactions)
            .select("user_id")
        )

        df = df.join(
            valid_users,
            on="user_id",
            how="semi",
        )

        valid_items = (
            df.group_by("parent_asin")
            .agg(pl.len().alias("n"))
            .filter(pl.col("n") >= min_item_interactions)
            .select("parent_asin")
        )

        df = df.join(
            valid_items,
            on="parent_asin",
            how="semi",
        )

        print(f"K-core iteration {iteration}: {old_rows:,} -> {len(df):,} interactions")

        if len(df) == old_rows:
            break

    return df

