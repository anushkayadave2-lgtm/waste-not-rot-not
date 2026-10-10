import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def check_forecasting_data_readiness(df: pd.DataFrame) -> Dict:
    """
    Evaluates whether the provided dataset contains the required historical
    sales records, product identifiers, and valid transaction dates needed
    for demand forecasting.
    """
    if df is None or df.empty:
        return {
            "is_ready": False,
            "can_run_ml": False,
            "can_run_baseline": False,
            "product_id_col": None,
            "date_col": None,
            "qty_col": None,
            "num_unique_dates": 0,
            "num_records": 0,
            "missing_fields": ["Product Identifier (item_name/item_id)", "Sales Date (sale_date)", "Sales Quantity (qty_sold)"],
            "reason": "No store dataset is loaded in memory.",
        }

    # 1. Product Identifier check
    product_id_col = None
    if "item_name" in df.columns and df["item_name"].dropna().nunique() > 0:
        product_id_col = "item_name"
    elif "item_id" in df.columns and df["item_id"].dropna().nunique() > 0:
        product_id_col = "item_id"

    # 2. Date column check
    date_col = None
    valid_dates_count = 0
    unique_dates_count = 0
    date_candidates = ["sale_date", "date_sold", "transaction_date", "date"]
    for dc in date_candidates:
        if dc in df.columns:
            parsed = pd.to_datetime(df[dc], errors="coerce").dropna()
            if not parsed.empty:
                date_col = dc
                valid_dates_count = len(parsed)
                unique_dates_count = parsed.dt.date.nunique()
                break

    # 3. Sales quantity check
    qty_col = None
    qty_candidates = ["qty_sold", "quantity_sold", "units_sold", "sales_qty", "daily_sales_avg", "sales"]
    for qc in qty_candidates:
        if qc in df.columns:
            numeric_q = pd.to_numeric(df[qc], errors="coerce").dropna()
            if not numeric_q.empty:
                qty_col = qc
                break

    # Build missing fields list
    missing = []
    if product_id_col is None:
        missing.append("Product Identifier (`item_name` or `item_id`)")
    if date_col is None:
        missing.append("Historical Sale Date (`sale_date`)")
    if qty_col is None:
        missing.append("Sales Quantity Sold (`qty_sold`)")

    # Assess ML and baseline capability
    can_run_ml = False
    can_run_baseline = False
    reason = ""

    if missing:
        reason = f"Missing required fields: {', '.join(missing)}."
    elif unique_dates_count < 3:
        can_run_baseline = True
        can_run_ml = False
        reason = (
            f"Dataset has only {unique_dates_count} distinct sales date(s). "
            "Holdout evaluation and ML time-series regression require at least 7 distinct transaction dates."
        )
    elif unique_dates_count < 7 or valid_dates_count < 14:
        can_run_baseline = True
        can_run_ml = False
        reason = (
            f"Dataset has {unique_dates_count} dates ({valid_dates_count} records). "
            "Sufficient for recent moving average baseline, but at least 7 dates are needed for holdout ML training."
        )
    else:
        can_run_baseline = True
        can_run_ml = True
        reason = f"Full historical sales ready ({unique_dates_count} transaction dates across {valid_dates_count} records)."

    return {
        "is_ready": (product_id_col is not None and qty_col is not None),
        "can_run_ml": can_run_ml,
        "can_run_baseline": can_run_baseline,
        "product_id_col": product_id_col,
        "date_col": date_col,
        "qty_col": qty_col,
        "num_unique_dates": unique_dates_count,
        "num_records": valid_dates_count,
        "missing_fields": missing,
        "reason": reason,
    }


def generate_sample_sales_history(days: int = 21, num_items: int = 8) -> pd.DataFrame:
    """
    Generates a realistic multi-day daily sales history dataset for demonstration
    and rigorous ML forecasting validation.
    """
    np.random.seed(42)
    today = datetime.today().date()
    start_date = today - timedelta(days=days)

    catalog = [
        {"item_id": "SKU-1001", "item_name": "Organic Spinach", "category": "Produce", "base_daily": 14.0, "unit_price": 3.99, "stock_qty": 35},
        {"item_id": "SKU-1002", "item_name": "Whole Milk 1L", "category": "Dairy & Eggs", "base_daily": 22.0, "unit_price": 2.89, "stock_qty": 60},
        {"item_id": "SKU-1003", "item_name": "Sourdough Bread", "category": "Bakery", "base_daily": 16.0, "unit_price": 4.50, "stock_qty": 25},
        {"item_id": "SKU-1004", "item_name": "Avocados 4pk", "category": "Produce", "base_daily": 11.0, "unit_price": 5.99, "stock_qty": 40},
        {"item_id": "SKU-1005", "item_name": "Greek Yogurt 500g", "category": "Dairy & Eggs", "base_daily": 13.0, "unit_price": 3.49, "stock_qty": 30},
        {"item_id": "SKU-1006", "item_name": "Chicken Breast 500g", "category": "Meat & Seafood", "base_daily": 18.0, "unit_price": 8.99, "stock_qty": 20},
        {"item_id": "SKU-1007", "item_name": "Bananas 1kg", "category": "Produce", "base_daily": 25.0, "unit_price": 1.79, "stock_qty": 80},
        {"item_id": "SKU-1008", "item_name": "Free-Range Eggs 12pk", "category": "Dairy & Eggs", "base_daily": 15.0, "unit_price": 4.99, "stock_qty": 18},
    ][:num_items]

    records = []
    current_date = start_date

    while current_date < today:
        weekday = current_date.weekday()
        # Realistic retail day-of-week demand multipliers (Saturday/Sunday boost, Tuesday dip)
        if weekday == 5:  # Saturday
            day_multiplier = 1.35
        elif weekday == 6:  # Sunday
            day_multiplier = 1.25
        elif weekday == 1:  # Tuesday
            day_multiplier = 0.85
        elif weekday == 4:  # Friday
            day_multiplier = 1.15
        else:
            day_multiplier = 1.0

        for item in catalog:
            noise = float(np.random.normal(0, item["base_daily"] * 0.12))
            qty = max(1, int(round(item["base_daily"] * day_multiplier + noise)))

            records.append({
                "item_id": item["item_id"],
                "item_name": item["item_name"],
                "category": item["category"],
                "sale_date": current_date.strftime("%Y-%m-%d"),
                "qty_sold": qty,
                "unit_price": item["unit_price"],
                "stock_qty": item["stock_qty"],
                "reorder_point": 15,
            })

        current_date += timedelta(days=1)

    return pd.DataFrame(records)


def aggregate_daily_sales(df: pd.DataFrame, id_col: str, date_col: str, qty_col: str) -> pd.DataFrame:
    """
    Aggregates historical sales transactions by product and day.
    Produces a clean, standardized time series table for model training.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    clean_df = df.copy()
    clean_df["_parsed_date"] = pd.to_datetime(clean_df[date_col], errors="coerce").dt.date
    clean_df["_numeric_qty"] = pd.to_numeric(clean_df[qty_col], errors="coerce")

    # Filter out missing records
    clean_df = clean_df.dropna(subset=["_parsed_date", "_numeric_qty", id_col])
    clean_df = clean_df[clean_df["_numeric_qty"] >= 0]

    if clean_df.empty:
        return pd.DataFrame()

    # Preserve metadata attributes (item_name, category, unit_price, stock_qty)
    meta_cols = {}
    for col in ["item_name", "category", "unit_price", "stock_qty"]:
        if col in clean_df.columns and col != id_col:
            meta_cols[col] = "first"

    agg_dict = {"_numeric_qty": "sum"}
    agg_dict.update(meta_cols)

    daily_agg = clean_df.groupby([id_col, "_parsed_date"], as_index=False).agg(agg_dict)
    daily_agg = daily_agg.rename(columns={"_parsed_date": "sale_date", "_numeric_qty": "qty_sold"})
    daily_agg = daily_agg.sort_values(["sale_date", id_col]).reset_index(drop=True)

    return daily_agg


def build_forecasting_features(daily_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Constructs time-series feature engineering matrix including calendar
    patterns, weekday cycles, and historical product lag sales.
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame()

    feat_df = daily_df.copy()
    feat_df["sale_date"] = pd.to_datetime(feat_df["sale_date"])

    # Calendar features
    feat_df["day_of_week"] = feat_df["sale_date"].dt.dayofweek
    feat_df["is_weekend"] = (feat_df["day_of_week"] >= 5).astype(int)
    feat_df["day_of_month"] = feat_df["sale_date"].dt.day

    # Categorical Product Encoding
    feat_df["product_code"] = feat_df[id_col].astype("category").cat.codes

    # Sort to compute lag and rolling averages safely per product
    feat_df = feat_df.sort_values([id_col, "sale_date"]).reset_index(drop=True)

    # Lags & rolling demand per product
    feat_df["lag_1"] = feat_df.groupby(id_col)["qty_sold"].shift(1)
    feat_df["lag_7"] = feat_df.groupby(id_col)["qty_sold"].shift(7)
    feat_df["rolling_mean_3"] = feat_df.groupby(id_col)["qty_sold"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    # Fill NaN lags with product historical mean
    item_means = feat_df.groupby(id_col)["qty_sold"].transform("mean")
    feat_df["lag_1"] = feat_df["lag_1"].fillna(item_means)
    feat_df["lag_7"] = feat_df["lag_7"].fillna(item_means)
    feat_df["rolling_mean_3"] = feat_df["rolling_mean_3"].fillna(item_means)

    return feat_df


def evaluate_models_on_holdout(
    daily_df: pd.DataFrame,
    id_col: str,
    test_ratio: float = 0.25,
) -> Dict:
    """
    Evaluates both a Baseline (Recent Average Daily Sales) and a Scikit-Learn
    RandomForest Regressor on chronological holdout data.
    Computes MAE, RMSE, and WAPE to provide verified, empirical accuracy comparisons.
    """
    if daily_df is None or daily_df.empty:
        return {"evaluated": False, "reason": "No daily sales data available."}

    feat_df = build_forecasting_features(daily_df, id_col)
    unique_dates = sorted(feat_df["sale_date"].unique())

    if len(unique_dates) < 4:
        return {
            "evaluated": False,
            "reason": f"Only {len(unique_dates)} distinct dates available. Need at least 4 dates for a holdout split.",
        }

    # Chronological Holdout Split (train on past, test on recent)
    split_idx = max(2, int(len(unique_dates) * (1 - test_ratio)))
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]

    if not test_dates:
        test_dates = unique_dates[-1:]
        train_dates = unique_dates[:-1]

    train_df = feat_df[feat_df["sale_date"].isin(train_dates)].copy()
    test_df = feat_df[feat_df["sale_date"].isin(test_dates)].copy()

    if train_df.empty or test_df.empty:
        return {"evaluated": False, "reason": "Insufficient train/test records after split."}

    # 1. Baseline Model: Recent Average Daily Sales per product
    baseline_item_means = train_df.groupby(id_col)["qty_sold"].mean().to_dict()
    global_mean = train_df["qty_sold"].mean()
    test_df["pred_baseline"] = test_df[id_col].map(baseline_item_means).fillna(global_mean).round(1)

    # 2. Scikit-Learn Machine Learning Model: RandomForestRegressor
    feature_cols = ["product_code", "day_of_week", "is_weekend", "day_of_month", "lag_1", "rolling_mean_3"]
    if "lag_7" in test_df.columns:
        feature_cols.append("lag_7")

    ml_model = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42)
    ml_model.fit(train_df[feature_cols], train_df["qty_sold"])

    raw_ml_preds = ml_model.predict(test_df[feature_cols])
    test_df["pred_ml"] = np.maximum(0.0, np.round(raw_ml_preds, 1))

    # 3. Calculate Holdout Evaluation Metrics
    y_true = test_df["qty_sold"].values
    y_base = test_df["pred_baseline"].values
    y_ml = test_df["pred_ml"].values

    mae_base = float(mean_absolute_error(y_true, y_base))
    mae_ml = float(mean_absolute_error(y_true, y_ml))

    rmse_base = float(np.sqrt(mean_squared_error(y_true, y_base)))
    rmse_ml = float(np.sqrt(mean_squared_error(y_true, y_ml)))

    total_actual = float(np.sum(y_true))
    wape_base = float((np.sum(np.abs(y_true - y_base)) / max(total_actual, 1.0)) * 100)
    wape_ml = float((np.sum(np.abs(y_true - y_ml)) / max(total_actual, 1.0)) * 100)

    # Relative improvement
    if mae_base > 0:
        error_reduction_pct = ((mae_base - mae_ml) / mae_base) * 100
    else:
        error_reduction_pct = 0.0

    # Clean holdout comparison table
    holdout_summary = test_df.groupby(test_df["sale_date"].dt.strftime("%Y-%m-%d"), as_index=False).agg({
        "qty_sold": "sum",
        "pred_baseline": "sum",
        "pred_ml": "sum",
    }).rename(columns={
        "sale_date": "holdout_date",
        "qty_sold": "actual_sales",
        "pred_baseline": "baseline_predicted",
        "pred_ml": "ml_predicted",
    })

    return {
        "evaluated": True,
        "train_dates_count": len(train_dates),
        "holdout_dates_count": len(test_dates),
        "holdout_start": min(test_dates).strftime("%Y-%m-%d"),
        "holdout_end": max(test_dates).strftime("%Y-%m-%d"),
        "total_holdout_records": len(test_df),
        "baseline_mae": round(mae_base, 2),
        "baseline_rmse": round(rmse_base, 2),
        "baseline_wape": round(wape_base, 1),
        "ml_mae": round(mae_ml, 2),
        "ml_rmse": round(rmse_ml, 2),
        "ml_wape": round(wape_ml, 1),
        "ml_improved": (mae_ml < mae_base),
        "error_reduction_pct": round(error_reduction_pct, 1),
        "trained_ml_model": ml_model,
        "feature_cols": feature_cols,
        "holdout_comparison_df": holdout_summary,
    }


def generate_future_forecast(
    daily_df: pd.DataFrame,
    id_col: str,
    horizon_days: int = 7,
    eval_dict: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Projects daily product sales over the user-selected forecast period (e.g. next 7 days).
    Keeps predicted numbers completely separate from actual historical records.
    Produces:
    1. daily_forecast_df (Detailed daily forecast by product and date)
    2. summary_forecast_df (Total horizon demand vs. on-hand stock comparison)
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    feat_df = build_forecasting_features(daily_df, id_col)
    latest_date = pd.to_datetime(feat_df["sale_date"]).max().date()
    products = feat_df[id_col].unique()

    # Product Metadata Mapping
    meta_info = {}
    for pid in products:
        item_rows = feat_df[feat_df[id_col] == pid]
        name = str(item_rows["item_name"].iloc[0]) if "item_name" in item_rows.columns else str(pid)
        cat = str(item_rows["category"].iloc[0]) if "category" in item_rows.columns else "General"
        stock = float(item_rows["stock_qty"].iloc[0]) if "stock_qty" in item_rows.columns and pd.notnull(item_rows["stock_qty"].iloc[0]) else None
        price = float(item_rows["unit_price"].iloc[0]) if "unit_price" in item_rows.columns and pd.notnull(item_rows["unit_price"].iloc[0]) else None
        mean_sales = float(item_rows["qty_sold"].mean()) if not item_rows.empty else 5.0
        meta_info[pid] = {
            "item_name": name,
            "category": cat,
            "stock_qty": stock,
            "unit_price": price,
            "mean_daily_sales": max(0.5, mean_sales),
        }

    # Model training on full available history
    ml_model = None
    feature_cols = ["product_code", "day_of_week", "is_weekend", "day_of_month", "lag_1", "rolling_mean_3"]
    if "lag_7" in feat_df.columns:
        feature_cols.append("lag_7")

    try:
        ml_model = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=42)
        ml_model.fit(feat_df[feature_cols], feat_df["qty_sold"])
    except Exception:
        ml_model = None

    use_ml_as_primary = bool(eval_dict and eval_dict.get("ml_improved", False))

    daily_predictions = []
    future_dates = [latest_date + timedelta(days=i) for i in range(1, horizon_days + 1)]

    for f_date in future_dates:
        weekday = f_date.weekday()
        is_weekend = int(weekday >= 5)
        day_of_month = f_date.day
        day_name = f_date.strftime("%A")

        for pid in products:
            meta = meta_info[pid]
            base_pred = round(meta["mean_daily_sales"], 1)

            # Weekend slight seasonal adjustment for baseline
            if is_weekend:
                base_pred = round(base_pred * 1.15, 1)

            ml_pred = base_pred
            if ml_model is not None:
                p_code = feat_df[feat_df[id_col] == pid]["product_code"].iloc[0]
                feat_dict = {
                    "product_code": [p_code],
                    "day_of_week": [weekday],
                    "is_weekend": [is_weekend],
                    "day_of_month": [day_of_month],
                    "lag_1": [base_pred],
                    "rolling_mean_3": [base_pred],
                }
                if "lag_7" in feature_cols:
                    feat_dict["lag_7"] = [base_pred]

                feat_df_pred = pd.DataFrame(feat_dict)
                try:
                    pred_val = float(ml_model.predict(feat_df_pred)[0])
                    ml_pred = max(0.0, round(pred_val, 1))
                except Exception:
                    ml_pred = base_pred

            recommended = ml_pred if use_ml_as_primary else base_pred

            daily_predictions.append({
                "item_id": str(pid),
                "item_name": meta["item_name"],
                "category": meta["category"],
                "forecast_date": f_date.strftime("%Y-%m-%d"),
                "day_of_week": day_name,
                "predicted_sales_baseline": base_pred,
                "predicted_sales_ml": ml_pred,
                "recommended_forecast": recommended,
                "record_type": "Projected Forecast",
            })

    daily_forecast_df = pd.DataFrame(daily_predictions)

    # Summary forecast by product across full horizon
    summary_list = []
    for pid in products:
        meta = meta_info[pid]
        item_daily = daily_forecast_df[daily_forecast_df["item_id"] == str(pid)]
        tot_base = round(float(item_daily["predicted_sales_baseline"].sum()), 1)
        tot_ml = round(float(item_daily["predicted_sales_ml"].sum()), 1)
        tot_rec = round(float(item_daily["recommended_forecast"].sum()), 1)
        stock = meta["stock_qty"]

        balance = round(stock - tot_rec, 1) if stock is not None else None
        stockout_risk = (stock is not None and tot_rec > stock)
        stockout_gap = max(0, int(round(tot_rec - stock))) if (stock is not None and tot_rec > stock) else 0
        overstock = max(0, int(round(stock - tot_rec))) if (stock is not None and stock > tot_rec) else 0

        summary_list.append({
            "item_id": str(pid),
            "item_name": meta["item_name"],
            "category": meta["category"],
            "forecast_horizon_days": horizon_days,
            "current_stock": int(stock) if stock is not None else "N/A",
            "projected_demand_baseline": tot_base,
            "projected_demand_ml": tot_ml,
            "recommended_total_demand": tot_rec,
            "projected_stock_balance": balance if balance is not None else "N/A",
            "stockout_risk": "⚠️ HIGH RISK" if stockout_risk else "🟢 SAFE STOCK",
            "projected_deficit_units": stockout_gap,
            "projected_overstock_units": overstock,
        })

    summary_forecast_df = pd.DataFrame(summary_list)
    return daily_forecast_df, summary_forecast_df


def create_holdout_evaluation_chart(eval_dict: Dict) -> Optional[go.Figure]:
    """
    Generates a Plotly comparison chart of actual sales vs. Baseline vs. ML predictions
    across the chronological holdout validation window.
    """
    if not eval_dict or not eval_dict.get("evaluated", False):
        return None

    comp_df = eval_dict.get("holdout_comparison_df")
    if comp_df is None or comp_df.empty:
        return None

    fig = go.Figure()

    # Actual historical holdout line
    fig.add_trace(
        go.Scatter(
            x=comp_df["holdout_date"],
            y=comp_df["actual_sales"],
            mode="lines+markers",
            name="Actual Holdout Sales",
            line=dict(color="#0F172A", width=3),
            marker=dict(size=8, color="#0F172A"),
            hovertemplate="<b>Date: %{x}</b><br>Actual Sales: %{y:,} units<extra></extra>",
        )
    )

    # Baseline prediction line
    fig.add_trace(
        go.Scatter(
            x=comp_df["holdout_date"],
            y=comp_df["baseline_predicted"],
            mode="lines+markers",
            name="Baseline (Recent Mean)",
            line=dict(color="#D97706", width=2, dash="dash"),
            marker=dict(size=6, color="#D97706"),
            hovertemplate="<b>Date: %{x}</b><br>Baseline Forecast: %{y:,} units<extra></extra>",
        )
    )

    # Scikit-Learn ML prediction line
    fig.add_trace(
        go.Scatter(
            x=comp_df["holdout_date"],
            y=comp_df["ml_predicted"],
            mode="lines+markers",
            name="Scikit-Learn ML Model",
            line=dict(color="#16A34A", width=2.5),
            marker=dict(size=7, color="#15803D"),
            hovertemplate="<b>Date: %{x}</b><br>ML Model Forecast: %{y:,} units<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text="<b>Holdout Validation: Actual vs. Baseline vs. ML Forecast</b>",
            font=dict(size=14, color="#0F172A"),
        ),
        xaxis=dict(title="Holdout Evaluation Date", gridcolor="#E2E8F0"),
        yaxis=dict(title="Total Units Sold", gridcolor="#E2E8F0"),
        margin=dict(l=10, r=10, t=40, b=20),
        height=340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_future_timeline_chart(daily_forecast_df: pd.DataFrame, selected_product: str = "All Products") -> Optional[go.Figure]:
    """
    Generates a Plotly timeline chart showing daily projected sales over the forecast horizon.
    """
    if daily_forecast_df is None or daily_forecast_df.empty:
        return None

    df_plot = daily_forecast_df.copy()
    if selected_product != "All Products" and "item_name" in df_plot.columns:
        df_plot = df_plot[df_plot["item_name"] == selected_product]

    timeline_df = df_plot.groupby("forecast_date", as_index=False).agg({
        "predicted_sales_baseline": "sum",
        "predicted_sales_ml": "sum",
        "recommended_forecast": "sum",
    })

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=timeline_df["forecast_date"],
            y=timeline_df["recommended_forecast"],
            name="Projected Daily Demand",
            marker=dict(color="#16A34A", line=dict(color="#15803D", width=1)),
            text=timeline_df["recommended_forecast"].round(1),
            textposition="auto",
            hovertemplate="<b>Date: %{x}</b><br>Projected Demand: %{y:,} units<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timeline_df["forecast_date"],
            y=timeline_df["predicted_sales_baseline"],
            mode="lines+markers",
            name="Baseline Average",
            line=dict(color="#F59E0B", width=2, dash="dot"),
            hovertemplate="<b>Baseline:</b> %{y:,} units<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(
            text=f"<b>Future Demand Projection Timeline ({selected_product})</b>",
            font=dict(size=14, color="#0F172A"),
        ),
        xaxis=dict(title="Forecast Date", gridcolor="#E2E8F0"),
        yaxis=dict(title="Projected Units", gridcolor="#E2E8F0"),
        margin=dict(l=10, r=10, t=40, b=20),
        height=330,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_stock_vs_demand_comparison_chart(summary_forecast_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Generates a Plotly comparison bar chart showing on-hand stock vs. projected demand
    to visibly identify impending stockout shortages or excessive overstocks.
    """
    if summary_forecast_df is None or summary_forecast_df.empty:
        return None

    # Filter out rows where stock is N/A
    valid_df = summary_forecast_df[summary_forecast_df["current_stock"] != "N/A"].copy()
    if valid_df.empty:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=valid_df["item_name"],
            y=valid_df["current_stock"],
            name="Current Stock On Hand",
            marker=dict(color="#64748B"),
            hovertemplate="<b>%{x}</b><br>Current Stock: %{y:,} units<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            x=valid_df["item_name"],
            y=valid_df["recommended_total_demand"],
            name="Projected Total Demand",
            marker=dict(color="#16A34A"),
            hovertemplate="<b>%{x}</b><br>Projected Demand: %{y:,} units<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title=dict(
            text="<b>Current Stock vs. Projected Demand by Product</b>",
            font=dict(size=14, color="#0F172A"),
        ),
        xaxis=dict(title=None, tickangle=-30),
        yaxis=dict(title="Units", gridcolor="#E2E8F0"),
        margin=dict(l=10, r=10, t=40, b=40),
        height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig
