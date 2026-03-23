# src/csv_eda_gemini_sync.py
# CSV EDA + Gemini (text + vision) — SYNC ONLY
# + Auto datetime detection + time trend charts (monthly/daily)

from __future__ import annotations

import io
import os
import tempfile
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image

try:
    import google.generativeai as genai
except Exception:
    genai = None


GEMINI_MODEL = "gemini-2.5-flash"


def init_gemini_sync() -> tuple[bool, str]:
    if genai is None:
        return False, "google-generativeai not installed."

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return False, "GOOGLE_API_KEY not set."

    try:
        genai.configure(api_key=api_key)
        _ = genai.GenerativeModel(GEMINI_MODEL)
        return True, f"Gemini ready: {GEMINI_MODEL}"
    except Exception as e:
        return False, f"Gemini init failed: {e}"


def _save_fig(fig) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    path = tmp.name
    tmp.close()

    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


def cleanup_files(paths: List[str]) -> None:
    for p in paths or []:
        try:
            os.remove(p)
        except Exception:
            pass


def df_context_string(df: pd.DataFrame, max_rows: int = 5) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    schema = buf.getvalue()

    preview = df.head(max_rows).to_markdown(index=False)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing_info = "No missing values." if missing.empty else missing.to_string()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        describe = df[numeric_cols].describe().to_markdown()
    else:
        describe = "No numeric columns."

    return f"""
### Schema

### Preview
{preview}

### Missing
{missing_info}

### Numeric Describe
{describe}
"""


def _detect_datetime_cols(df: pd.DataFrame, object_cols: List[str]) -> List[str]:
    """
    Detect datetime-like columns among object columns by sampling and parsing.
    A column is considered datetime-like if >=60% of sampled rows parse as datetime.
    """
    dt_cols: List[str] = []
    if df.empty or not object_cols:
        return dt_cols

    sample_n = min(len(df), 200)
    sample_df = df[object_cols].head(sample_n)

    for col in object_cols:
        s = sample_df[col].astype(str)

        parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
        ratio = float(parsed.notna().mean()) if len(parsed) else 0.0

        if ratio >= 0.60:
            dt_cols.append(col)

    return dt_cols


def generate_visuals(df: pd.DataFrame):
    visuals: List[Tuple[str, str]] = []
    tmp_files: List[str] = []

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    object_cols = df.select_dtypes(include="object").columns.tolist()

    # ✅ detect time columns
    datetime_cols = _detect_datetime_cols(df, object_cols)

    # -----------------------
    # Correlation heatmap
    # -----------------------
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            ax=ax
        )
        ax.set_title("Correlation Heatmap")

        p = _save_fig(fig)
        visuals.append(("Correlation Heatmap", p))
        tmp_files.append(p)

    # -----------------------
    # Pairplot
    # -----------------------
    if len(numeric_cols) >= 3:
        plot_df = df[numeric_cols[:5]].dropna()
        if len(plot_df) > 2000:
            plot_df = plot_df.sample(2000, random_state=42)

        g = sns.pairplot(plot_df)
        g.fig.suptitle("Pairplot of Numeric Features", y=1.02)

        p = _save_fig(g.fig)
        visuals.append(("Pairplot", p))
        tmp_files.append(p)

    # -----------------------
    # Numeric distributions
    # -----------------------
    for col in numeric_cols[:3]:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.violinplot(data=df, y=col, ax=ax)
        ax.set_title(f"Violin Plot - {col}")
        p = _save_fig(fig)
        visuals.append((f"Violin Plot - {col}", p))
        tmp_files.append(p)

        fig, ax = plt.subplots(figsize=(8, 6))
        df[col].dropna().plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Histogram - {col}")
        p = _save_fig(fig)
        visuals.append((f"Histogram - {col}", p))
        tmp_files.append(p)

    # -----------------------
    # Categorical top values (exclude datetime-like cols)
    # -----------------------
    cat_cols = [c for c in object_cols if c not in datetime_cols]
    for col in cat_cols[:3]:
        counts = df[col].value_counts(dropna=True).head(10)
        if len(counts) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Top 10 Values - {col}")
        plt.xticks(rotation=45, ha="right")

        p = _save_fig(fig)
        visuals.append((f"Top 10 - {col}", p))
        tmp_files.append(p)

    # -----------------------
    # ✅ Time trend charts (auto)
    # -----------------------
    for col in datetime_cols[:2]:
        dates = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        valid = dates.dropna()
        if len(valid) < 5:
            continue

        span_days = int((valid.max() - valid.min()).days) if len(valid) else 0

        if span_days > 120:
            # monthly
            grp = valid.dt.to_period("M").value_counts().sort_index()
            title = f"Monthly Trend - {col}"
            xlabel = "Month"
        else:
            # daily
            grp = valid.dt.date.value_counts().sort_index()
            title = f"Daily Trend - {col}"
            xlabel = "Date"

        fig, ax = plt.subplots(figsize=(10, 6))
        grp.plot(kind="line", marker="o", ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")

        p = _save_fig(fig)
        visuals.append((title, p))
        tmp_files.append(p)

    return visuals, tmp_files


def _gemini_text_sync(prompt: str):
    if genai is None:
        return "Gemini SDK not installed."

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        res = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=600,
                temperature=0.3
            )
        )
        return res.text.strip() if getattr(res, "text", None) else "⚠️ Gemini response empty."
    except Exception as e:
        return f"❌ Gemini error: {e}"


def _gemini_vision_sync(title: str, img_path: str):
    if genai is None:
        return "Gemini SDK not installed."

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        img = Image.open(img_path)

        prompt = f"""
Explain the insights from this chart.

Chart title: {title}

Return 3-6 concise bullet insights.
"""

        res = model.generate_content(
            [prompt, img],
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=250,
                temperature=0.2
            )
        )
        return res.text.strip() if getattr(res, "text", None) else "⚠️ Gemini response empty."
    except Exception as e:
        return f"❌ Gemini vision error: {e}"


def run_csv_eda_sync(df: pd.DataFrame, gemini_available: bool):
    context_md = df_context_string(df)
    visuals, tmp_files = generate_visuals(df)

    if gemini_available:
        plan = _gemini_text_sync(
            "You are a senior data analyst.\n"
            "Suggest a concise analysis plan.\n\n"
            + context_md
        )

        summary = _gemini_text_sync(
            "Summarize the key insights from the dataset.\n\n"
            + context_md
        )

        vision_notes = []
        for title, path in visuals:
            explanation = _gemini_vision_sync(title, path)
            vision_notes.append((title, explanation))
    else:
        plan = "Gemini not available."
        summary = "Gemini not available."
        vision_notes = [(t, "Gemini not available.") for t, _ in visuals]

    return {
        "context_md": context_md,
        "visuals": visuals,
        "tmp_files": tmp_files,
        "plan": plan,
        "summary": summary,
        "vision_notes": vision_notes,
        "gemini_available": bool(gemini_available),
        "gemini_model": GEMINI_MODEL,
    }
