"""AI Analysis Engine - generates insights, explanations, and recommendations"""
import ast
import operator
import pandas as pd
import numpy as np


_ALLOWED_ATTRS = {
    "head", "tail", "nlargest", "nsmallest", "sort_values", "groupby", "agg",
    "mean", "median", "sum", "min", "max", "count", "std", "var", "describe",
    "value_counts", "unique", "nunique", "isna", "notna", "fillna", "dropna",
    "drop_duplicates", "astype", "round", "pct_change", "quantile", "idxmax",
    "idxmin", "corr", "loc", "iloc", "shape", "columns", "index", "dtypes",
    "reset_index", "rename", "select_dtypes", "melt", "pivot_table", "crosstab",
    "sum", "abs", "clip", "diff", "cumsum", "sample", "first", "last",
    "to_string", "tolist", "keys", "get", "values", "size", "empty", "max",
    "min", "astype", "str", "dt", "iat", "at", "is_monotonic_increasing",
}

_ALLOWED_FUNCS = {
    "len", "str", "int", "float", "sum", "min", "max", "abs", "round",
    "sorted", "list", "set", "dict", "bool", "tuple", "pd", "np", "range",
}

_ALLOWED_NODES = (
    ast.Expression, ast.Module, ast.Expr, ast.Call, ast.Attribute, ast.Name,
    ast.Load, ast.Constant, ast.BinOp, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.Compare, ast.Eq,
    ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.List, ast.Tuple, ast.Dict,
    ast.Subscript, ast.Index, ast.Slice, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp,
    ast.Not, ast.USub, ast.UAdd, ast.IfExp, ast.keyword, ast.Starred,
)


_BLOCKED_TERMS = {
    "os", "sys", "subprocess", "importlib", "eval", "exec", "open", "__import__",
    "to_csv", "to_excel", "to_json", "to_feather", "to_parquet", "to_pickle", "to_sql",
    "remove", "unlink", "rmdir", "system", "popen", "read_csv", "read_excel",
}


def validate_expression(code):
    """AST whitelist & blacklist validation. Returns (ok: bool, error: str)."""
    if not code.strip():
        return False, "Expression is empty."

    # Pre-check for any explicit forbidden terms
    code_lower = code.lower()
    for term in _BLOCKED_TERMS:
        if term in code_lower:
            return False, f"Blocked forbidden operation or system module: '{term}'."

    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        return False, f"Invalid syntax: {e}"

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False, f"Blocked construct: {type(node).__name__}"
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_FUNCS:
                    return False, f"Blocked function call: '{node.func.id}()'. Only pandas/numpy-safe operations allowed."
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in _ALLOWED_ATTRS:
                    return False, f"Blocked method: '.{node.func.attr}()'. Operation not in whitelist."
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in ("pd", "np"):
                continue
            if node.attr not in _ALLOWED_ATTRS and not (isinstance(node.value, ast.Attribute) and node.value.attr in ("str", "dt")):
                if node.attr in ("loc", "iloc", "values", "index", "columns", "shape", "dtypes", "size", "empty", "max", "min", "sum", "count", "mean", "median", "first", "last"):
                    continue
                return False, f"Blocked attribute access: '.{node.attr}'. Operation not in whitelist."
        elif isinstance(node, ast.Name):
            if node.id not in _ALLOWED_FUNCS and node.id != "_df":
                return False, f"Blocked identifier: '{node.id}'. Only '_df' and safe builtins allowed."
    return True, ""


def safe_eval_df(code, df):
    """Evaluate an AST-validated pandas expression against df (max ~5MB of result)."""
    ok, err = validate_expression(code)
    if not ok:
        raise ValueError(err)
    env = {
        "_df": df, "pd": pd, "np": np,
        "len": len, "str": str, "int": int, "float": float,
        "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
        "sorted": sorted, "list": list, "set": set, "dict": dict,
        "bool": bool, "tuple": tuple, "range": range,
    }
    try:
        tree = compile(ast.parse(code, mode="eval"), "<sandbox>", "eval")
        result = eval(tree, {"__builtins__": {}}, env)
        if isinstance(result, pd.Series) or isinstance(result, pd.DataFrame):
            if result.memory_usage(deep=True).sum() > 5 * 1024 * 1024:
                raise ValueError("Result too large — refine your query (limit rows/columns).")
            return result
        if isinstance(result, (list, dict, tuple, np.ndarray)):
            return result
        return result
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Execution error: {e}")


def translate_nl_to_code(query, df, col_types):
    """Heuristic NL → pandas translation for common query intents."""
    q = query.lower()
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "date"]
    all_cols = df.columns.tolist()

    def find_col(keywords):
        for k in keywords:
            for c in all_cols:
                if k in c.lower():
                    return c
        return None

    n = "10"
    import re
    m = re.search(r"top\s*(\d+)", q) or re.search(r"(\d+)\s*(?:top|highest)", q)
    if m:
        n = m.group(1)

    if any(w in q for w in ["average", "mean", "avg"]):
        col = find_col(num_cols) or (num_cols[0] if num_cols else None)
        return f"_df['{col}'].mean()" if col else None, f"Mean of {col}"
    if any(w in q for w in ["total", "sum of"]):
        col = find_col(num_cols) or (num_cols[0] if num_cols else None)
        return f"_df['{col}'].sum()" if col else None, f"Sum of {col}"
    if any(w in q for w in ["top", "highest", "maximum"]):
        col = find_col(num_cols) or (num_cols[0] if num_cols else None)
        return f"_df.nlargest({n}, '{col}')" if col else None, f"Top {n} by {col}"
    if any(w in q for w in ["lowest", "minimum", "bottom"]):
        col = find_col(num_cols) or (num_cols[0] if num_cols else None)
        return f"_df.nsmallest({n}, '{col}')" if col else None, f"Bottom {n} by {col}"
    if any(w in q for w in ["missing", "null", "nan"]):
        return "_df.isna().sum()", "Missing values per column"
    if any(w in q for w in ["correlation", "correlate"]):
        return f"_df[{num_cols}].corr()" if len(num_cols) >= 2 else None, "Correlation matrix"
    if any(w in q for w in ["describe", "summary", "overview", "stats"]):
        return "_df.describe()", "Statistical summary"
    if any(w in q for w in ["unique", "distinct"]):
        col = find_col(cat_cols) or (cat_cols[0] if cat_cols else None)
        return f"_df['{col}'].value_counts()" if col else None, f"Value counts of {col}"
    if any(w in q for w in ["count", "rows"]):
        return "len(_df)", "Row count"
    return None, None


def analyze_dataset(df, col_types):
    stats = {
        "rows": len(df),
        "cols": len(df.columns),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "num_cols": sum(1 for v in col_types.values() if v == "numeric"),
        "cat_cols": sum(1 for v in col_types.values() if v == "categorical"),
        "date_cols": sum(1 for v in col_types.values() if v == "date"),
        "text_cols": sum(1 for v in col_types.values() if v == "text"),
        "bool_cols": sum(1 for v in col_types.values() if v == "boolean"),
        "missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2) if len(df) > 0 else 0,
        "duplicate_pct": round(df.duplicated().sum() / len(df) * 100, 2) if len(df) > 0 else 0,
    }
    quality = "excellent"
    if stats["missing_pct"] > 20 or stats["duplicate_pct"] > 20:
        quality = "poor"
    elif stats["missing_pct"] > 5 or stats["duplicate_pct"] > 5:
        quality = "fair"
    stats["quality"] = quality
    return stats


def data_quality_score(df, col_types):
    """Returns a 0-100 quality score with breakdown."""
    n = len(df)
    if n == 0:
        return 0, {}
    missing_pct = df.isnull().sum().sum() / (n * len(df.columns)) * 100
    dup_pct = df.duplicated().sum() / n * 100
    # Outlier pct across numeric cols
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    outlier_pct = 0
    if num_cols:
        outlier_counts = []
        for c in num_cols:
            s = df[c].dropna()
            if len(s) > 0 and s.std() > 0:
                outlier_counts.append(int((abs((s - s.mean()) / s.std()) > 3).sum()))
        total_outliers = sum(outlier_counts)
        outlier_pct = total_outliers / (n * len(num_cols)) * 100 if num_cols else 0

    missing_score = max(0, 40 - missing_pct * 2)
    dup_score = max(0, 30 - dup_pct * 1.5)
    outlier_score = max(0, 30 - outlier_pct * 3)
    total = round(missing_score + dup_score + outlier_score)

    breakdown = {
        "Missing Values": round(missing_score, 1),
        "Duplicates": round(dup_score, 1),
        "Outliers": round(outlier_score, 1),
        "missing_pct": round(missing_pct, 2),
        "dup_pct": round(dup_pct, 2),
        "outlier_pct": round(outlier_pct, 2),
    }
    return min(total, 100), breakdown


def smart_chart_recommendations(df, col_types, max_charts=16):
    """
    AI-driven chart recommendations based on:
    - Correlation between numeric columns
    - Cardinality of categorical columns
    - Variance/spread of numeric columns
    - Presence of date columns for time-series
    Returns list of dicts: {chart_type, x, y, color, reason, priority}
    """
    recs = []
    num_cols = [c for c, t in col_types.items() if t == "numeric"]
    cat_cols = [c for c, t in col_types.items() if t == "categorical"]
    date_cols = [c for c, t in col_types.items() if t == "date"]

    # Sample for performance on large datasets
    df_sample = df.sample(min(5000, len(df)), random_state=42) if len(df) > 5000 else df

    # ── 1. Correlation-based scatter plots (top pairs) ──
    if len(num_cols) >= 2:
        try:
            corr = df_sample[num_cols].corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = upper.unstack().dropna().sort_values(ascending=False)
            for (col_a, col_b), corr_val in pairs.head(4).items():
                if corr_val > 0.1:
                    color_col = cat_cols[0] if cat_cols else None
                    reason = (
                        f"Strong correlation ({corr_val:.2f}) between {col_a} and {col_b} — "
                        "scatter plot reveals relationship direction and outliers."
                        if corr_val > 0.5
                        else f"Moderate correlation ({corr_val:.2f}) — scatter plot helps detect patterns."
                    )
                    recs.append({
                        "chart_type": "Scatter Plot",
                        "x": col_a, "y": col_b,
                        "color": color_col,
                        "reason": reason,
                        "priority": corr_val * 10,
                    })
        except Exception:
            pass

    # ── 2. High-variance numeric → histogram ──
    if num_cols:
        variances = {c: df_sample[c].dropna().var() for c in num_cols if len(df_sample[c].dropna()) > 0}
        sorted_var = sorted(variances.items(), key=lambda x: x[1], reverse=True)
        for col, var in sorted_var[:3]:
            skew = df_sample[col].dropna().skew()
            direction = "right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "symmetric")
            recs.append({
                "chart_type": "Histogram",
                "x": col, "y": None,
                "color": None,
                "reason": f"High variance in {col} — distribution is {direction}. Histogram shows spread and shape.",
                "priority": min(var / max(variances.values()) * 7, 7) if variances else 3,
            })

    # ── 3. Categorical columns → bar charts ──
    for col in cat_cols:
        n_unique = df_sample[col].nunique()
        top_val = df_sample[col].value_counts().index[0] if len(df_sample[col].dropna()) > 0 else "N/A"
        top_pct = df_sample[col].value_counts().iloc[0] / len(df_sample[col].dropna()) * 100 if len(df_sample[col].dropna()) > 0 else 0
        if n_unique <= 30:
            recs.append({
                "chart_type": "Bar Chart",
                "x": col, "y": None,
                "color": None,
                "reason": f"{col} has {n_unique} categories. Most common: '{top_val}' ({top_pct:.1f}%). Bar chart shows frequency distribution.",
                "priority": 8 if n_unique <= 10 else 5,
            })
        # Pie for low cardinality
        if n_unique <= 8:
            recs.append({
                "chart_type": "Pie Chart",
                "x": col, "y": None,
                "color": None,
                "reason": f"{col} has only {n_unique} categories — pie chart shows proportional share of each.",
                "priority": 6,
            })

    # ── 4. Categorical vs Numeric → Box/Violin plots ──
    if cat_cols and num_cols:
        best_cat = max(cat_cols, key=lambda c: df_sample[c].nunique() if df_sample[c].nunique() <= 15 else 0)
        for num_col in num_cols[:2]:
            recs.append({
                "chart_type": "Box Plot",
                "x": best_cat, "y": num_col,
                "color": best_cat,
                "reason": f"Box plot reveals how {num_col} varies across {best_cat} categories — shows median, spread, and outliers per group.",
                "priority": 7,
            })
            recs.append({
                "chart_type": "Violin Plot",
                "x": best_cat, "y": num_col,
                "color": best_cat,
                "reason": f"Violin plot shows full distribution shape of {num_col} per {best_cat} category — richer than a box plot.",
                "priority": 6,
            })

    # ── 5. Time-series → line charts ──
    if date_cols and num_cols:
        date_col = date_cols[0]
        for num_col in num_cols[:3]:
            recs.append({
                "chart_type": "Line Chart",
                "x": date_col, "y": num_col,
                "color": None,
                "reason": f"Date column detected — line chart tracks {num_col} over time to reveal trends, seasonality, or sudden shifts.",
                "priority": 9,
            })

    # ── 6. Grouped bar (cat x num aggregation) ──
    if cat_cols and num_cols:
        for cat_col in cat_cols[:2]:
            for num_col in num_cols[:2]:
                recs.append({
                    "chart_type": "Grouped Bar",
                    "x": cat_col, "y": num_col,
                    "color": None,
                    "reason": f"Average {num_col} per {cat_col} category — grouped bar chart makes comparisons easy.",
                    "priority": 7.5,
                })

    # ── 7. Treemap for hierarchical categories ──
    if cat_cols and num_cols:
        recs.append({
            "chart_type": "Treemap",
            "x": cat_cols[0], "y": num_cols[0],
            "color": None,
            "reason": f"Treemap shows proportional area for each {cat_cols[0]} weighted by {num_cols[0]} — great for part-of-whole analysis.",
            "priority": 5,
        })

    # ── 8. Area chart for numeric trends ──
    if num_cols:
        recs.append({
            "chart_type": "Area Chart",
            "x": None, "y": num_cols[0],
            "color": None,
            "reason": f"Area chart emphasises cumulative magnitude of {num_cols[0]} across records.",
            "priority": 4,
        })

    # ── 9. Correlation heatmap ──
    if len(num_cols) >= 3:
        recs.append({
            "chart_type": "Correlation Heatmap",
            "x": None, "y": None,
            "color": None,
            "reason": f"Heatmap of all {len(num_cols)} numeric columns — instantly reveals which pairs are strongly correlated.",
            "priority": 9,
        })

    # Sort by priority, deduplicate chart_type+x+y combos, return top N
    recs.sort(key=lambda r: r["priority"], reverse=True)
    seen = set()
    unique_recs = []
    for r in recs:
        key = (r["chart_type"], r["x"], r["y"])
        if key not in seen:
            seen.add(key)
            unique_recs.append(r)

    return unique_recs[:max_charts]


def column_insight(col_name, series, dtype):
    s = series.dropna()
    if len(s) == 0:
        return f"<strong>{col_name}</strong> has no non-null values and may need to be removed."
    if dtype == "numeric":
        skew = s.skew()
        outliers = int(((s - s.mean()).abs() > 2 * s.std()).sum())
        direction = "right-skewed" if skew > 0.5 else ("left-skewed" if skew < -0.5 else "symmetric")
        return (
            f"<strong>{col_name}</strong> (numeric): {len(s)} values, range [{s.min():.2f}, {s.max():.2f}], "
            f"mean={s.mean():.2f}, median={s.median():.2f}. Distribution is {direction} "
            f"with {outliers} potential outliers."
        )
    elif dtype == "categorical":
        top = s.value_counts().index[0] if len(s.value_counts()) > 0 else "N/A"
        top_pct = round(s.value_counts().iloc[0] / len(s) * 100, 1) if len(s) > 0 else 0
        return (
            f"<strong>{col_name}</strong> (categorical): {s.nunique()} unique categories. "
            f"Most common is <strong>{top}</strong> ({top_pct}% of data). "
            f"{'High cardinality - consider grouping.' if s.nunique() > 20 else ''}"
        )
    elif dtype == "date":
        return (
            f"<strong>{col_name}</strong> (date): spans from {s.min().date()} to {s.max().date()}, "
            f"covering {s.dt.year.nunique()} years. Useful for time-based analysis."
        )
    elif dtype == "text":
        avg_len = s.str.len().mean()
        return (
            f"<strong>{col_name}</strong> (text): {len(s)} entries, average {avg_len:.0f} characters. "
            f"{'Short text field - good for labels.' if avg_len < 50 else 'Long text - suitable for NLP analysis.'}"
        )
    elif dtype == "boolean":
        true_pct = round(s.sum() / len(s) * 100, 1) if len(s) > 0 else 0
        return f"<strong>{col_name}</strong> (boolean): {true_pct}% True, {100 - true_pct}% False."
    return f"<strong>{col_name}</strong>: type={dtype}, {len(s)} values."


def chart_explanation(chart_type, x_col, y_col, df_slice):
    if chart_type == "Bar Chart":
        top_vals = df_slice.head(5)
        return f"This bar chart shows the distribution of <strong>{x_col}</strong>. The highest value is <strong>{top_vals.iloc[0][x_col] if x_col in top_vals.columns else ''}</strong>, suggesting this category dominates the dataset."
    elif chart_type == "Line Chart":
        return f"This line chart tracks <strong>{y_col or x_col}</strong> over <strong>{x_col}</strong>. Look for upward or downward trends — a rising line indicates growth, while a falling line suggests decline."
    elif chart_type == "Pie Chart":
        return f"This pie chart breaks down <strong>{x_col}</strong> into proportions. Larger slices represent categories with greater representation in the dataset."
    elif chart_type == "Histogram":
        return f"This histogram shows the frequency distribution of <strong>{x_col}</strong>. The shape reveals whether the data is symmetric, skewed, or has multiple peaks."
    elif chart_type == "Scatter Plot":
        return f"This scatter plot shows the relationship between <strong>{x_col}</strong> and <strong>{y_col}</strong>. A clear pattern suggests correlation — upward trends indicate positive correlation."
    elif chart_type == "Box Plot":
        return f"This box plot shows the spread of <strong>{y_col or x_col}</strong>. The box contains the middle 50% of data; points beyond the whiskers are potential outliers."
    elif chart_type == "Correlation Heatmap":
        return "This heatmap visualizes correlations between numeric columns. Dark red indicates strong positive correlation, dark blue indicates strong negative correlation."
    elif chart_type == "Area Chart":
        return f"This area chart emphasizes the magnitude of change in <strong>{y_col or x_col}</strong> over <strong>{x_col}</strong>. The filled area makes trends more visually apparent."
    else:
        return f"This {chart_type} visualizes <strong>{x_col}</strong>" + (f" against <strong>{y_col}</strong>" if y_col else "") + ". Use it to identify patterns, outliers, and relationships in your data."


def correlation_insight(col_a, col_b, corr_value, method):
    strength = "very strong" if abs(corr_value) > 0.8 else "strong" if abs(corr_value) > 0.6 else "moderate" if abs(corr_value) > 0.4 else "weak"
    direction = "positive" if corr_value > 0 else "negative"
    if abs(corr_value) > 0.7:
        biz = f"When <strong>{col_a}</strong> increases, <strong>{col_b}</strong> {'also increases' if corr_value > 0 else 'decreases'} significantly."
    else:
        biz = f"The relationship between <strong>{col_a}</strong> and <strong>{col_b}</strong> is {strength}."
    return f"The {method} correlation between <strong>{col_a}</strong> and <strong>{col_b}</strong> is <strong>{corr_value:.3f}</strong> ({strength} {direction}). {biz} This suggests {'a meaningful relationship worth investigating further.' if abs(corr_value) > 0.5 else 'a limited linear relationship between these variables.'}"


def important_columns(df, col_types, top_n=5):
    scores = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        score = 0
        dtype = col_types.get(col, "text")
        if dtype == "numeric":
            cv = s.std() / s.mean() if s.mean() != 0 else 0
            score = cv * s.nunique()
        elif dtype == "categorical":
            entropy = -(s.value_counts(normalize=True) * np.log(s.value_counts(normalize=True) + 1e-10)).sum()
            score = entropy * np.log(s.nunique() + 1)
        elif dtype == "date":
            score = s.dt.year.nunique() * 10
        elif dtype == "text":
            score = s.str.len().mean() * s.nunique() / 100
        scores.append((col, round(score, 2), dtype))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def cleaning_recommendations(df, col_types):
    recs = []
    missing_map = df.isnull().sum().to_dict()
    dup_count = df.duplicated().sum()
    for col, miss in missing_map.items():
        if miss > 0:
            pct = miss / len(df) * 100
            dtype = col_types.get(col, "text")
            if pct > 50:
                recs.append(f"🔴 <strong>{col}</strong>: {miss} missing ({pct:.1f}%) — consider removing this column")
            elif dtype == "numeric":
                recs.append(f"🟡 <strong>{col}</strong>: {miss} missing ({pct:.1f}%) — fill with <strong>median</strong> ({df[col].median():.2f})")
            elif dtype == "categorical":
                mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "mode"
                recs.append(f"🟡 <strong>{col}</strong>: {miss} missing ({pct:.1f}%) — fill with most common value (<strong>{mode_val}</strong>)")
            else:
                recs.append(f"🟡 <strong>{col}</strong>: {miss} missing ({pct:.1f}%) — consider removing these rows")
    if dup_count > 0:
        recs.append(f"🟡 <strong>Duplicate rows</strong>: {dup_count} found ({dup_count/len(df)*100:.1f}%) — remove to avoid skewed analysis")
    if not recs:
        recs.append("✅ Dataset is clean — no missing values or duplicates detected.")
    return recs


def row_insight(df, row_idx, col_types):
    row = df.iloc[row_idx]
    parts = [f"<strong>Row {row_idx}</strong> analysis:"]
    for col in df.columns[:8]:
        val = row[col]
        dtype = col_types.get(col, "text")
        if pd.isna(val):
            parts.append(f"- {col}: <strong>missing</strong>")
        elif dtype == "numeric":
            s = df[col].dropna()
            z = (val - s.mean()) / s.std() if s.std() > 0 else 0
            tag = "🔥 HIGH" if abs(z) > 2 else "✅ normal"
            parts.append(f"- {col}: <strong>{val:.2f}</strong> ({tag}, z={z:.2f})")
        elif dtype == "categorical":
            pct = (df[col] == val).sum() / len(df) * 100
            parts.append(f"- {col}: <strong>{val}</strong> ({pct:.1f}% of records)")
        elif dtype == "date":
            parts.append(f"- {col}: <strong>{val}</strong>")
        else:
            parts.append(f"- {col}: <strong>{val}</strong>")
    return "\n".join(parts)


def row_comparison(df, row_indices, col_types):
    if len(row_indices) < 2:
        return "Select at least 2 rows for comparison."
    rows = [df.iloc[i] for i in row_indices]
    parts = [f"<strong>Comparing rows {', '.join(map(str, row_indices))}:</strong>"]
    for col in df.columns[:10]:
        vals = [r[col] for r in rows]
        dtype = col_types.get(col, "text")
        if len(set(str(v) for v in vals)) == 1:
            parts.append(f"- {col}: all rows = <strong>{vals[0]}</strong> (consistent)")
        elif dtype == "numeric":
            diff = max(vals) - min(vals)
            s = df[col].dropna()
            parts.append(f"- {col}: ranges from <strong>{min(vals):.2f}</strong> to <strong>{max(vals):.2f}</strong> (Δ={diff:.2f}, {diff/s.std():.1f}σ)" if s.std() > 0 else f"- {col}: ranges from <strong>{min(vals):.2f}</strong> to <strong>{max(vals):.2f}</strong>")
        else:
            unique_vals = set(str(v) for v in vals)
            parts.append(f"- {col}: <strong>{' vs '.join(str(v) for v in vals)}</strong> ({len(unique_vals)} unique)")
    return "\n".join(parts)


def analyze_crosstab(row_col, col_col, df):
    from scipy import stats as scipy_stats
    ct = pd.crosstab(df[row_col], df[col_col])
    chi2, p, dof, expected = scipy_stats.chi2_contingency(ct) if ct.size > 0 else (0, 1, 0, ct)
    cramer_v = np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1))) if len(df) * (min(ct.shape) - 1) > 0 else 0
    return {
        "chi2": round(chi2, 2),
        "p_value": round(p, 6),
        "cramers_v": round(cramer_v, 4),
        "significant": p < 0.05,
        "association": "strong" if cramer_v > 0.5 else "moderate" if cramer_v > 0.3 else "weak",
        "insight": f"There is {'a <strong>significant</strong>' if p < 0.05 else '<strong>no significant</strong>'} association between <strong>{row_col}</strong> and <strong>{col_col}</strong> "
                   f"(χ²={chi2:.2f}, p={p:.4f}, Cramér\'s V={cramer_v:.3f}). "
                   f"The relationship is <strong>{('strong' if cramer_v > 0.5 else 'moderate' if cramer_v > 0.3 else 'weak')}</strong>."
    }


def forecast_insight(col, slope, intercept, r2, future_periods):
    direction = "increase" if slope > 0 else "decrease"
    confidence = "high" if r2 > 0.7 else "moderate" if r2 > 0.4 else "low"
    return (
        f"📈 <strong>{col} Forecast</strong>: Based on historical trend, values are expected to <strong>{direction}</strong> "
        f"at a rate of {abs(slope):.4f} per period. The model has <strong>{confidence} confidence</strong> (R²={r2:.3f}). "
        f"Projected values for the next {future_periods} periods are shown above."
    )
