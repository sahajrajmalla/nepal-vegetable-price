#!/usr/bin/env python3
"""
combine_data.py — Merge Raw CSVs + Add Commodity Categories
=============================================================

Combines the two raw Kalimati datasets into a single unified CSV with:
    • Standardised schema: [Commodity, Date, Unit, Minimum, Maximum, Average]
    • Commodity_Category column (Vegetables, Fruits, Spices, Fish, etc.)
    • Commodity_Group column (finer sub-group, e.g., Leafy Vegetables)
    • Cleaned date formats (YYYY-MM-DD)
    • Removed duplicates
    • Sorted chronologically

Saves:
    data/processed/full_data.csv           — Complete merged dataset
    data/processed/commodity_mapping.csv   — Category & group reference table

Author : Sahaj Raj Malla
Created: 2025
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT.parent / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT.parent / "data" / "processed"


# ═════════════════════════════════════════════════════════════════════════════
# Commodity Category Mapping (keyword-based, case-insensitive)
# ═════════════════════════════════════════════════════════════════════════════

# Category → Group → list of keyword patterns
CATEGORY_RULES = {
    "Fruits": {
        "Citrus Fruits": [
            "orange", "lemon", "lime", "mandarin", "kinnow",
            "sweet lime", "sweet orange",
        ],
        "Tropical Fruits": [
            "banana", "mango", "papaya", "pineapple", "litchi",
            "jack fruit", "guava", "pomegranate", "tamarind",
            "sugarcane", "sarifa",
        ],
        "Temperate Fruits": [
            "apple", "pear", "strawberry", "kiwi", "grapes",
            "avocado", "amla", "mombin",
        ],
        "Melons": [
            "water melon", "watermelon", "musk melon",
        ],
        "Berries & Others": [
            "tree tomato", "bakula",
        ],
    },
    "Leafy Vegetables": {
        "Leafy Greens": [
            "spinach", "mustard leaf", "brd leaf mustard",
            "cress leaf", "fenugreek leaf", "lettuce",
            "coriander green", "mint", "parseley", "celery",
            "fennel leaf",
        ],
    },
    "Root & Tuber Vegetables": {
        "Root Vegetables": [
            "potato", "carrot", "raddish", "radish", "turnip",
            "sweet potato", "sugarbeet", "yam", "arum",
            "knolkhol",
        ],
    },
    "Solanaceous Vegetables": {
        "Tomatoes": ["tomato"],
        "Brinjals & Peppers": [
            "brinjal", "capsicum",
        ],
    },
    "Cucurbits": {
        "Gourds & Squash": [
            "bitter gourd", "bottle gourd", "smooth gourd",
            "snake gourd", "sponge gourd", "pumpkin",
            "squash", "christophine", "pointed gourd",
            "cucumber",
        ],
    },
    "Bulb & Allium Vegetables": {
        "Onions & Garlic": [
            "onion", "garlic", "clive",
        ],
    },
    "Legume Vegetables": {
        "Beans & Peas": [
            "french bean", "green peas", "cow pea", "cowpea",
            "soyabean", "sword bean", "okara",
        ],
    },
    "Brassica Vegetables": {
        "Cabbage & Cauliflower": [
            "cabbage", "cauli", "brocauli", "red cabbbage",
        ],
    },
    "Spices & Condiments": {
        "Fresh Spices": [
            "chilli", "ginger",
        ],
        "Dried Spices": [
            "chilli dry",
        ],
    },
    "Mushrooms": {
        "Mushrooms": [
            "mushroom",
        ],
    },
    "Fish": {
        "Fresh Fish": [
            "fish",
        ],
    },
    "Other": {
        "Processed": [
            "tofu", "gundruk",
        ],
        "Shoots & Others": [
            "bamboo shoot", "neuro", "asparagus", "drumstick",
            "barela", "bauhania",
        ],
        "Grains & Cereals": [
            "maize",
        ],
    },
}


def classify_commodity(name: str) -> tuple:
    """
    Classify a commodity name into (Category, Group).

    Uses keyword matching against the CATEGORY_RULES dictionary.
    Returns ('Uncategorised', 'Uncategorised') if no match found.
    """
    name_lower = name.lower().strip()

    for category, groups in CATEGORY_RULES.items():
        for group, keywords in groups.items():
            for kw in keywords:
                if kw in name_lower:
                    return category, group

    return "Uncategorised", "Uncategorised"


def load_and_merge() -> pd.DataFrame:
    """
    Load both raw CSV files and merge into a single DataFrame.

    Handles:
    - First CSV: has SN column + header row
    - Second CSV: no header row, dates in M/D/YYYY format
    """
    COLS = ["Commodity", "Date", "Unit", "Minimum", "Maximum", "Average"]

    frames = []

    # ── File 1: 2013–2021 (has header + SN column) ──
    f1 = DATA_RAW / "Kalimati Tarkari Prices from June 2013 to May 2021.csv"
    if f1.exists():
        df1 = pd.read_csv(f1, encoding="utf-8")
        df1.columns = df1.columns.str.strip()
        if "SN" in df1.columns:
            df1 = df1.drop(columns=["SN"])
        print(f"  File 1: {len(df1):,} rows loaded ({f1.name})")
        frames.append(df1)
    else:
        print(f"  WARNING: {f1} not found")

    # ── File 2: 2021–2023 (no header) ──
    f2 = DATA_RAW / "Kalimati Tarkari Prices from May 2021 to September 2023.csv"
    if f2.exists():
        df2 = pd.read_csv(f2, encoding="utf-8", header=None, names=COLS)
        print(f"  File 2: {len(df2):,} rows loaded ({f2.name})")
        frames.append(df2)
    else:
        print(f"  WARNING: {f2} not found")

    if not frames:
        raise RuntimeError("No raw data files found!")

    combined = pd.concat(frames, ignore_index=True)
    return combined


def clean_and_categorise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged data and add category columns.

    Steps:
    1. Clean price columns (remove 'Rs ' prefix)
    2. Parse dates to uniform YYYY-MM-DD
    3. Normalise units (KG → Kg)
    4. Add Commodity_Category and Commodity_Group
    5. Remove exact duplicates
    6. Sort by Date, Commodity
    """
    # Clean price columns
    for col in ["Minimum", "Maximum", "Average"]:
        df[col] = (
            df[col].astype(str)
            .str.replace(r"[Rr][Ss]\s*", "", regex=True)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)

    # Normalise units
    df["Unit"] = df["Unit"].str.strip().str.title()  # KG → Kg

    # Normalise commodity names (strip whitespace)
    df["Commodity"] = df["Commodity"].str.strip()

    # ── Add category columns ──
    categories = df["Commodity"].apply(classify_commodity)
    df["Commodity_Category"] = categories.apply(lambda x: x[0])
    df["Commodity_Group"] = categories.apply(lambda x: x[1])

    # Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["Commodity", "Date", "Minimum", "Maximum", "Average"])
    after = len(df)
    if before > after:
        print(f"  Removed {before - after:,} duplicate rows")

    # Sort
    df = df.sort_values(["Date", "Commodity"]).reset_index(drop=True)

    return df


def print_category_summary(df: pd.DataFrame) -> None:
    """Print a summary of commodities by category."""
    print("\n" + "=" * 70)
    print("  COMMODITY CATEGORY SUMMARY")
    print("=" * 70)

    summary = (
        df.groupby(["Commodity_Category", "Commodity_Group"])["Commodity"]
        .nunique()
        .reset_index(name="Unique_Commodities")
    )

    for cat in sorted(summary["Commodity_Category"].unique()):
        cat_rows = summary[summary["Commodity_Category"] == cat]
        total = cat_rows["Unique_Commodities"].sum()
        print(f"\n  {cat} ({total} commodities)")
        for _, row in cat_rows.iterrows():
            print(f"    └─ {row['Commodity_Group']}: {row['Unique_Commodities']}")

    uncategorised = df[df["Commodity_Category"] == "Uncategorised"]["Commodity"].unique()
    if len(uncategorised) > 0:
        print(f"\n  ⚠ Uncategorised ({len(uncategorised)}): {list(uncategorised)}")

    print(f"\n  Total: {df['Commodity'].nunique()} unique commodities, "
          f"{len(df):,} records, "
          f"{df['Date'].min().date()} → {df['Date'].max().date()}")


def create_commodity_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """Create a reference table mapping each commodity to its category/group."""
    mapping = (
        df.groupby("Commodity")
        .agg(
            Category=("Commodity_Category", "first"),
            Group=("Commodity_Group", "first"),
            Unit=("Unit", "first"),
            Record_Count=("Date", "count"),
            Date_Min=("Date", "min"),
            Date_Max=("Date", "max"),
            Avg_Price_Mean=("Average", "mean"),
        )
        .reset_index()
        .sort_values(["Category", "Group", "Commodity"])
    )
    mapping["Date_Min"] = mapping["Date_Min"].dt.date
    mapping["Date_Max"] = mapping["Date_Max"].dt.date
    mapping["Avg_Price_Mean"] = mapping["Avg_Price_Mean"].round(2)
    return mapping


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  COMBINING RAW KALIMATI DATA + COMMODITY CATEGORISATION           ║")
    print("╚" + "═" * 68 + "╝")

    # Load and merge
    print("\n── Loading raw CSVs ──")
    df = load_and_merge()
    print(f"  Combined: {len(df):,} total rows")

    # Clean and categorise
    print("\n── Cleaning & categorising ──")
    df = clean_and_categorise(df)

    # Summary
    print_category_summary(df)

    # Save
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    out_path = DATA_PROCESSED / "full_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved full dataset: {out_path} ({len(df):,} rows)")

    # Save commodity mapping reference
    mapping = create_commodity_mapping(df)
    mapping_path = DATA_PROCESSED / "commodity_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    print(f"✓ Saved commodity mapping: {mapping_path} ({len(mapping)} commodities)")

    # Also save a Parquet version for fast loading
    parquet_path = DATA_PROCESSED / "full_data.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"✓ Saved Parquet version: {parquet_path}")

    print("\n" + "═" * 70)
    print("  DONE — Use data/processed/full_data.csv for analysis")
    print("═" * 70)


if __name__ == "__main__":
    main()
