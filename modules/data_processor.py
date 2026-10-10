import io
import pandas as pd


def generate_sample_data(num_items: int = 25) -> pd.DataFrame:
    """Generates realistic sample grocery inventory data for testing and demonstration."""
    import numpy as np
    from datetime import datetime, timedelta

    np.random.seed(42)
    categories = ["Produce", "Dairy & Eggs", "Bakery", "Meat & Seafood", "Pantry"]
    items_by_cat = {
        "Produce": ["Organic Spinach", "Avocados", "Strawberries", "Bananas", "Tomatoes"],
        "Dairy & Eggs": ["Whole Milk 1L", "Greek Yogurt 500g", "Cheddar Cheese", "Free-Range Eggs", "Oat Milk 1L"],
        "Bakery": ["Sourdough Bread", "Croissants 4pk", "Whole Wheat Loaf", "Bagels 6pk", "Muffins"],
        "Meat & Seafood": ["Chicken Breast 500g", "Salmon Fillet", "Ground Beef", "Turkey Slice", "Tuna Steaks"],
        "Pantry": ["Canned Beans", "Pasta 500g", "Olive Oil 750ml", "Rice 1kg", "Tomato Sauce"],
    }

    data = []
    today = datetime.today().date()
    item_id = 1001
    counter = 0

    for category in categories:
        items = items_by_cat[category]
        for name in items:
            counter += 1
            # Inject realistic distribution: a couple expired, a couple nearing expiry, a couple low stock
            if counter == 3:  # Expired item
                days_to_expiry = -2
                stock_qty = 14
                reorder_point = 15
            elif counter == 4:  # Expired item
                days_to_expiry = -1
                stock_qty = 8
                reorder_point = 10
            elif counter == 7:  # Urgent expiry (1 day)
                days_to_expiry = 1
                stock_qty = 25
                reorder_point = 20
            elif counter == 8:  # Urgent expiry (3 days)
                days_to_expiry = 3
                stock_qty = 40
                reorder_point = 15
            elif counter == 12:  # Out of stock
                days_to_expiry = 12
                stock_qty = 0
                reorder_point = 20
            elif counter == 16:  # Low stock
                days_to_expiry = 14
                stock_qty = 5
                reorder_point = 18
            else:
                days_to_expiry = int(np.random.randint(4, 25))
                stock_qty = int(np.random.randint(15, 120))
                reorder_point = int(np.random.randint(10, 30))

            expiry_date = today + timedelta(days=days_to_expiry)
            unit_price = round(float(np.random.uniform(1.5, 18.0)), 2)
            daily_sales_avg = round(float(np.random.uniform(2.0, 15.0)), 1)
            qty_sold = int(daily_sales_avg * 7)
            # Recent transaction date within the past 14 days
            sale_date = (today - timedelta(days=int(counter % 10))).strftime("%Y-%m-%d")

            stock_cover_days = stock_qty / max(daily_sales_avg, 0.1)
            if days_to_expiry < 0:
                waste_risk = 100.0
            elif stock_cover_days > days_to_expiry:
                waste_risk = min(100.0, round(((stock_cover_days - days_to_expiry) / stock_cover_days) * 100, 1))
            else:
                waste_risk = round(float(np.random.uniform(0, 15)), 1)

            data.append(
                {
                    "item_id": f"SKU-{item_id}",
                    "item_name": name,
                    "category": category,
                    "stock_qty": stock_qty,
                    "unit": "units",
                    "unit_price": unit_price,
                    "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                    "days_to_expiry": days_to_expiry,
                    "qty_sold": qty_sold,
                    "sale_date": sale_date,
                    "daily_sales_avg": daily_sales_avg,
                    "reorder_point": reorder_point,
                    "waste_risk_pct": waste_risk,
                }
            )
            item_id += 1

    return pd.DataFrame(data)


def read_uploaded_file(uploaded_file):
    """
    Reads a CSV or Excel (.xlsx/.xls) file using Pandas and OpenPyXL.
    Returns (dataframe, error_message).
    """
    if uploaded_file is None:
        return None, "No file provided."

    filename = uploaded_file.name.lower()

    try:
        content = uploaded_file.read()
        file_bytes = io.BytesIO(content)
        
        if filename.endswith(".csv"):
            df = pd.read_csv(file_bytes)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_bytes, engine="openpyxl")
        else:
            return None, "Unsupported file format. Please upload a CSV (.csv) or Excel (.xlsx) file."
    except Exception as e:
        return None, f"Unable to parse file. The file may be corrupted or incorrectly formatted: {str(e)}"

    if df is None or df.empty:
        return None, "The uploaded file is empty or contains no records."

    return df, None


def combine_separate_files(df_inv: pd.DataFrame = None, df_sales: pd.DataFrame = None, df_expiry: pd.DataFrame = None) -> pd.DataFrame:
    """
    Combines separate uploaded DataFrames (Inventory, Sales, Expiry) into a unified master DataFrame.
    """
    dfs_to_merge = []
    if df_inv is not None and not df_inv.empty:
        dfs_to_merge.append(df_inv.copy())
    if df_sales is not None and not df_sales.empty:
        dfs_to_merge.append(df_sales.copy())
    if df_expiry is not None and not df_expiry.empty:
        dfs_to_merge.append(df_expiry.copy())

    if not dfs_to_merge:
        return pd.DataFrame()

    # Start with first available dataframe
    merged = dfs_to_merge[0]

    for next_df in dfs_to_merge[1:]:
        if "item_id" in merged.columns and "item_id" in next_df.columns:
            # Merge on item_id key
            cols_to_use = [c for c in next_df.columns if c not in merged.columns or c == "item_id"]
            merged = pd.merge(merged, next_df[cols_to_use], on="item_id", how="outer")
        else:
            # Fallback column concat/combine
            for col in next_df.columns:
                if col not in merged.columns:
                    merged[col] = next_df[col]

    return merged


def validate_dataframe(df: pd.DataFrame, file_label: str = "Dataset") -> tuple[bool, str]:
    """
    Validates that a DataFrame has rows and readable columns.
    """
    if df is None:
        return False, f"{file_label} is empty."

    if df.empty:
        return False, f"{file_label} contains 0 rows."

    if len(df.columns) == 0:
        return False, f"{file_label} has no valid column headers."

    return True, f"{file_label} validated successfully ({len(df)} rows, {len(df.columns)} columns)."


def generate_sample_template_csv(template_type: str = "single") -> bytes:
    """Generates a downloadable sample CSV template byte stream."""
    df = _create_template_dataframe(template_type)
    return df.to_csv(index=False).encode("utf-8")


def generate_sample_template_excel(template_type: str = "single") -> bytes:
    """Generates a downloadable sample Excel (.xlsx) template byte stream using OpenPyXL."""
    df = _create_template_dataframe(template_type)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="StoreDataTemplate")
    return output.getvalue()


def _create_template_dataframe(template_type: str) -> pd.DataFrame:
    """Internal helper creating clean un-hardcoded schema structure for templates."""
    if template_type == "inventory":
        return pd.DataFrame({
            "item_id": ["SKU-1001", "SKU-1002", "SKU-1003"],
            "item_name": ["Product A", "Product B", "Product C"],
            "category": ["Category 1", "Category 2", "Category 1"],
            "stock_qty": [50, 120, 15],
            "unit_price": [4.99, 2.50, 12.99],
            "reorder_point": [15, 30, 10]
        })
    elif template_type == "sales":
        return pd.DataFrame({
            "item_id": ["SKU-1001", "SKU-1002", "SKU-1003"],
            "daily_sales_avg": [8.5, 14.0, 3.2],
            "last_30_days_sales": [255, 420, 96]
        })
    elif template_type == "expiry":
        return pd.DataFrame({
            "item_id": ["SKU-1001", "SKU-1002", "SKU-1003"],
            "expiry_date": ["2026-10-15", "2026-10-25", "2026-10-11"],
            "days_to_expiry": [6, 16, 2]
        })
    else:  # Combined single file
        return pd.DataFrame({
            "item_id": ["SKU-1001", "SKU-1002", "SKU-1003"],
            "item_name": ["Product A", "Product B", "Product C"],
            "category": ["Produce", "Dairy", "Bakery"],
            "stock_qty": [45, 80, 12],
            "unit_price": [3.99, 5.49, 2.99],
            "expiry_date": ["2026-10-12", "2026-10-20", "2026-10-10"],
            "days_to_expiry": [3, 11, 1],
            "daily_sales_avg": [6.0, 10.5, 4.0],
            "reorder_point": [15, 20, 10],
            "waste_risk_pct": [35.0, 5.0, 75.0]
        })


def calculate_inventory_metrics(df: pd.DataFrame) -> dict:
    """Computes high-level inventory statistics safely from any DataFrame."""
    if df is None or df.empty:
        return {
            "total_skus": 0,
            "total_units": 0,
            "total_value": 0.0,
            "high_risk_items": 0,
            "expiring_soon_3d": 0,
        }

    total_skus = len(df)
    total_units = int(df["stock_qty"].sum()) if "stock_qty" in df.columns else 0
    
    if "stock_qty" in df.columns and "unit_price" in df.columns:
        total_value = float((df["stock_qty"] * df["unit_price"]).sum())
    else:
        total_value = 0.0

    expiring_soon = int((df["days_to_expiry"] <= 3).sum()) if "days_to_expiry" in df.columns else 0
    high_risk = int((df["waste_risk_pct"] > 40.0).sum()) if "waste_risk_pct" in df.columns else 0

    return {
        "total_skus": total_skus,
        "total_units": total_units,
        "total_value": round(total_value, 2),
        "high_risk_items": high_risk,
        "expiring_soon_3d": expiring_soon,
    }


def export_to_excel(df: pd.DataFrame) -> bytes:
    """Exports DataFrame to Excel bytes using OpenPyXL engine for user download."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Inventory_Report")
    return output.getvalue()


def calculate_dashboard_metrics(df: pd.DataFrame, low_stock_threshold: int = 10) -> dict:
    """
    Computes all essential store dashboard metrics from the active dataset.
    Strictly calculates values without hardcoded defaults and cleanly
    distinguishes unavailable metrics (None) from zero values.
    """
    from datetime import datetime

    if df is None or df.empty:
        return {
            "has_data": False,
            "distinct_products": None,
            "distinct_products_available": False,
            "total_stock_qty": None,
            "stock_qty_available": False,
            "units_compatible": True,
            "unit_label": "units",
            "unit_breakdown": {},
            "low_stock_count": None,
            "out_of_stock_count": None,
            "low_stock_available": False,
            "low_stock_threshold_label": "",
            "expired_count": None,
            "expiring_soon_3d": None,
            "expiring_soon_7d": None,
            "total_expiry_risk": None,
            "expiry_available": False,
            "total_qty_sold": None,
            "total_sales_revenue": None,
            "avg_daily_sales": None,
            "sales_record_count": None,
            "sales_available": False,
            "sales_date_min": None,
            "sales_date_max": None,
            "total_valuation": None,
            "valuation_available": False,
        }

    # 1. Distinct products
    distinct_products = None
    distinct_products_available = False
    if "item_id" in df.columns and df["item_id"].dropna().nunique() > 0:
        distinct_products = int(df["item_id"].dropna().nunique())
        distinct_products_available = True
    elif "item_name" in df.columns and df["item_name"].dropna().nunique() > 0:
        distinct_products = int(df["item_name"].dropna().nunique())
        distinct_products_available = True
    elif len(df) > 0:
        distinct_products = len(df)
        distinct_products_available = True

    # 2. Total stock quantity & unit compatibility
    total_stock_qty = None
    stock_qty_available = False
    units_compatible = True
    unit_label = "units"
    unit_breakdown = {}

    if "stock_qty" in df.columns:
        sq = pd.to_numeric(df["stock_qty"], errors="coerce")
        stock_qty_available = True
        total_stock_qty = float(sq.dropna().sum())

        if "unit" in df.columns:
            valid_units = df["unit"].dropna().astype(str).str.strip().str.lower()
            valid_units = valid_units[~valid_units.isin(["nan", "none", "", "null"])]
            unique_u = valid_units.unique()
            if len(unique_u) > 1:
                units_compatible = False
                unit_label = "Mixed Units"
                temp_df = pd.DataFrame({"qty": sq, "unit": valid_units}).dropna()
                if not temp_df.empty:
                    unit_breakdown = temp_df.groupby("unit")["qty"].sum().to_dict()
            elif len(unique_u) == 1:
                unit_label = unique_u[0]
                unit_breakdown = {unit_label: total_stock_qty}
            else:
                unit_label = "units"
                unit_breakdown = {"units": total_stock_qty}
        else:
            unit_label = "units"
            unit_breakdown = {"units": total_stock_qty}

    # 3. Products with low stock & out of stock
    low_stock_count = None
    out_of_stock_count = None
    low_stock_available = False
    low_stock_threshold_label = ""

    if "stock_qty" in df.columns:
        low_stock_available = True
        sq = pd.to_numeric(df["stock_qty"], errors="coerce").fillna(0)
        out_of_stock_count = int((sq == 0).sum())

        if "reorder_point" in df.columns:
            rp = pd.to_numeric(df["reorder_point"], errors="coerce").fillna(low_stock_threshold)
            is_low = (sq <= rp)
            low_stock_threshold_label = "Reorder Threshold"
        else:
            is_low = (sq <= low_stock_threshold)
            low_stock_threshold_label = f"<= {low_stock_threshold} units"

        low_stock_count = int(is_low.sum())

    # 4. Products nearing expiry or already expired
    expired_count = None
    expiring_soon_3d = None
    expiring_soon_7d = None
    total_expiry_risk = None
    expiry_available = False

    days_col = None
    if "days_to_expiry" in df.columns:
        days_col = pd.to_numeric(df["days_to_expiry"], errors="coerce")
        expiry_available = True
    elif "expiry_date" in df.columns:
        parsed_d = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date
        today = datetime.today().date()
        days_col = parsed_d.apply(lambda d: (d - today).days if pd.notnull(d) else np.nan)
        expiry_available = True

    if expiry_available and days_col is not None:
        expired_count = int((days_col < 0).sum())
        expiring_soon_3d = int(((days_col >= 0) & (days_col <= 3)).sum())
        expiring_soon_7d = int(((days_col > 3) & (days_col <= 7)).sum())
        total_expiry_risk = expired_count + expiring_soon_3d

    # 5. Historical sales summary
    sales_available = False
    total_qty_sold = None
    total_sales_revenue = None
    avg_daily_sales = None
    sales_record_count = None
    sales_date_min = None
    sales_date_max = None

    sales_qty_col = None
    if "qty_sold" in df.columns:
        sales_qty_col = "qty_sold"
    elif "daily_sales_avg" in df.columns:
        sales_qty_col = "daily_sales_avg"

    if sales_qty_col is not None or "sale_date" in df.columns:
        sales_available = True
        if sales_qty_col is not None:
            sq_sold = pd.to_numeric(df[sales_qty_col], errors="coerce").dropna()
            total_qty_sold = float(sq_sold.sum())
            avg_daily_sales = round(float(sq_sold.mean()), 1) if not sq_sold.empty else 0.0

            if "unit_price" in df.columns:
                prices = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
                total_sales_revenue = round(float((sq_sold * prices).sum()), 2)

        if "sale_date" in df.columns:
            s_dates = pd.to_datetime(df["sale_date"], errors="coerce").dropna()
            if not s_dates.empty:
                sales_date_min = s_dates.min().strftime("%Y-%m-%d")
                sales_date_max = s_dates.max().strftime("%Y-%m-%d")
                sales_record_count = len(s_dates)

    # 6. Inventory Valuation
    valuation_available = False
    total_valuation = None
    if "stock_qty" in df.columns and "unit_price" in df.columns:
        sq = pd.to_numeric(df["stock_qty"], errors="coerce").fillna(0)
        up = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)
        total_valuation = round(float((sq * up).sum()), 2)
        valuation_available = True

    return {
        "has_data": True,
        "distinct_products": distinct_products,
        "distinct_products_available": distinct_products_available,
        "total_stock_qty": total_stock_qty,
        "stock_qty_available": stock_qty_available,
        "units_compatible": units_compatible,
        "unit_label": unit_label,
        "unit_breakdown": unit_breakdown,
        "low_stock_count": low_stock_count,
        "out_of_stock_count": out_of_stock_count,
        "low_stock_available": low_stock_available,
        "low_stock_threshold_label": low_stock_threshold_label,
        "expired_count": expired_count,
        "expiring_soon_3d": expiring_soon_3d,
        "expiring_soon_7d": expiring_soon_7d,
        "total_expiry_risk": total_expiry_risk,
        "expiry_available": expiry_available,
        "total_qty_sold": total_qty_sold,
        "total_sales_revenue": total_sales_revenue,
        "avg_daily_sales": avg_daily_sales,
        "sales_record_count": sales_record_count,
        "sales_available": sales_available,
        "sales_date_min": sales_date_min,
        "sales_date_max": sales_date_max,
        "total_valuation": total_valuation,
        "valuation_available": valuation_available,
    }


def get_dashboard_alerts(df: pd.DataFrame, low_stock_threshold: int = 10, max_alerts: int = 8) -> list:
    """
    Generates a prioritized, concise list of actionable inventory alerts.
    Rank order:
    1. Already expired products (critical action needed)
    2. Expiring in 0-3 days (immediate markdown / donation)
    3. Out of stock products (immediate reorder)
    4. Low stock products (reorder warning)
    5. Expiring in 4-7 days (shelf-life notice)
    """
    from datetime import datetime

    if df is None or df.empty:
        return []

    alerts = []
    today = datetime.today().date()

    for _, row in df.iterrows():
        item_name = str(row.get("item_name", row.get("item_id", "Product")))
        category = str(row.get("category", "General"))
        stock = row.get("stock_qty")
        stock = float(stock) if pd.notnull(stock) else None

        # Expiry resolution
        days = row.get("days_to_expiry")
        if pd.isnull(days) and "expiry_date" in row and pd.notnull(row["expiry_date"]):
            try:
                parsed_date = pd.to_datetime(row["expiry_date"]).date()
                days = (parsed_date - today).days
            except Exception:
                days = None
        days = int(days) if pd.notnull(days) else None

        # Low stock resolution
        reorder_pt = row.get("reorder_point")
        reorder_pt = float(reorder_pt) if pd.notnull(reorder_pt) else float(low_stock_threshold)

        # 1. Already Expired
        if days is not None and days < 0:
            alerts.append({
                "severity_rank": 0,
                "severity": "EXPIRED",
                "badge": "🚨 ALREADY EXPIRED",
                "badge_bg": "#FEE2E2",
                "badge_color": "#DC2626",
                "border_color": "#EF4444",
                "item_name": item_name,
                "category": category,
                "message": f"Expired {abs(days)} day(s) ago ({row.get('expiry_date', 'Past Date')}). Current shelf stock: {int(stock) if stock is not None else 'N/A'}.",
                "action": "Remove from shelf immediately",
            })
        # 2. Urgent Expiry (0-3 days)
        elif days is not None and 0 <= days <= 3:
            day_text = "Today!" if days == 0 else f"in {days} day(s)"
            alerts.append({
                "severity_rank": 1,
                "severity": "URGENT_EXPIRY",
                "badge": "⚠️ EXPIRING SOON",
                "badge_bg": "#FFEDD5",
                "badge_color": "#EA580C",
                "border_color": "#F97316",
                "item_name": item_name,
                "category": category,
                "message": f"Expires {day_text} ({row.get('expiry_date', '')}). Stock: {int(stock) if stock is not None else 'N/A'} units.",
                "action": "Apply dynamic markdown or donate",
            })
        # 3. Out of stock
        elif stock is not None and stock == 0:
            alerts.append({
                "severity_rank": 2,
                "severity": "OUT_OF_STOCK",
                "badge": "📦 OUT OF STOCK",
                "badge_bg": "#FEE2E2",
                "badge_color": "#DC2626",
                "border_color": "#EF4444",
                "item_name": item_name,
                "category": category,
                "message": "Zero inventory units on hand. Product is currently stocked out.",
                "action": "Place replenishment order",
            })
        # 4. Low stock
        elif stock is not None and stock > 0 and stock <= reorder_pt:
            alerts.append({
                "severity_rank": 3,
                "severity": "LOW_STOCK",
                "badge": "🟡 LOW STOCK",
                "badge_bg": "#FEF3C7",
                "badge_color": "#D97706",
                "border_color": "#F59E0B",
                "item_name": item_name,
                "category": category,
                "message": f"Only {int(stock)} units remaining (Reorder threshold: {int(reorder_pt)}).",
                "action": "Review restock schedule",
            })
        # 5. Warning shelf life (4-7 days)
        elif days is not None and 4 <= days <= 7:
            alerts.append({
                "severity_rank": 4,
                "severity": "SHELF_WARNING",
                "badge": "🌱 SHELF LIFE NOTICE",
                "badge_bg": "#FEF9C3",
                "badge_color": "#CA8A04",
                "border_color": "#EAB308",
                "item_name": item_name,
                "category": category,
                "message": f"Expires in {days} days ({row.get('expiry_date', '')}). Stock: {int(stock) if stock is not None else 'N/A'} units.",
                "action": "Monitor sales velocity",
            })

    alerts.sort(key=lambda a: (a["severity_rank"], a["item_name"]))
    return alerts[:max_alerts]


def create_category_inventory_chart(df: pd.DataFrame):
    """Generates category-wise inventory breakdown bar chart using Plotly."""
    if df is None or df.empty or "category" not in df.columns:
        return None

    import plotly.graph_objects as go

    has_stock = "stock_qty" in df.columns
    if has_stock:
        cat_df = df.groupby("category", as_index=False)["stock_qty"].sum()
        cat_df = cat_df.sort_values("stock_qty", ascending=True)
        y_val = cat_df["category"]
        x_val = cat_df["stock_qty"]
        x_title = "Total Units in Stock"
    else:
        cat_df = df.groupby("category", as_index=False).size().rename(columns={"size": "count"})
        cat_df = cat_df.sort_values("count", ascending=True)
        y_val = cat_df["category"]
        x_val = cat_df["count"]
        x_title = "Distinct Products (SKUs)"

    fig = go.Figure(
        go.Bar(
            x=x_val,
            y=y_val,
            orientation="h",
            marker=dict(
                color="#16A34A",
                line=dict(color="#15803D", width=1),
            ),
            text=x_val,
            textposition="auto",
            hovertemplate="<b>%{y}</b><br>" + x_title + ": %{x:,}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(text="<b>Category-Wise Inventory Distribution</b>", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title=x_title, gridcolor="#E2E8F0"),
        yaxis=dict(title=None, tickfont=dict(size=12)),
        margin=dict(l=10, r=10, t=40, b=20),
        height=320,
    )
    return fig


def create_sales_trend_chart(df: pd.DataFrame, start_date=None, end_date=None):
    """Generates historical sales trend chart using Plotly."""
    if df is None or df.empty:
        return None

    import plotly.graph_objects as go

    if "sale_date" in df.columns:
        df_sales = df.dropna(subset=["sale_date"]).copy()
        df_sales["sale_date"] = pd.to_datetime(df_sales["sale_date"], errors="coerce")
        df_sales = df_sales.dropna(subset=["sale_date"])

        if start_date:
            df_sales = df_sales[df_sales["sale_date"].dt.date >= pd.to_datetime(start_date).date()]
        if end_date:
            df_sales = df_sales[df_sales["sale_date"].dt.date <= pd.to_datetime(end_date).date()]

        if df_sales.empty:
            return None

        qty_col = "qty_sold" if "qty_sold" in df_sales.columns else ("daily_sales_avg" if "daily_sales_avg" in df_sales.columns else None)
        if qty_col:
            trend_df = df_sales.groupby(df_sales["sale_date"].dt.strftime("%Y-%m-%d"), as_index=False)[qty_col].sum()
            trend_df = trend_df.sort_values("sale_date")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=trend_df["sale_date"],
                    y=trend_df[qty_col],
                    mode="lines+markers",
                    line=dict(color="#16A34A", width=3),
                    marker=dict(size=7, color="#15803D"),
                    fill="tozeroy",
                    fillcolor="rgba(34, 197, 94, 0.12)",
                    hovertemplate="<b>Date: %{x}</b><br>Units Sold: %{y:,}<extra></extra>",
                )
            )

            fig.update_layout(
                template="plotly_white",
                title=dict(text="<b>Historical Sales Volume Over Time</b>", font=dict(size=14, color="#0F172A")),
                xaxis=dict(title="Transaction Date", gridcolor="#E2E8F0"),
                yaxis=dict(title="Units Sold", gridcolor="#E2E8F0"),
                margin=dict(l=10, r=10, t=40, b=20),
                height=320,
            )
            return fig

    # Fallback when qty_sold exists without sale_date
    qty_col = "qty_sold" if "qty_sold" in df.columns else ("daily_sales_avg" if "daily_sales_avg" in df.columns else None)
    if qty_col and "item_name" in df.columns:
        top_items = df.groupby("item_name", as_index=False)[qty_col].sum()
        top_items = top_items.sort_values(qty_col, ascending=False).head(8)
        top_items = top_items.sort_values(qty_col, ascending=True)

        fig = go.Figure(
            go.Bar(
                x=top_items[qty_col],
                y=top_items["item_name"],
                orientation="h",
                marker=dict(color="#0D9488", line=dict(color="#0F766E", width=1)),
                text=top_items[qty_col],
                textposition="auto",
                hovertemplate="<b>%{y}</b><br>Units Sold: %{x:,}<extra></extra>",
            )
        )
        fig.update_layout(
            template="plotly_white",
            title=dict(text="<b>Top Products by Sales Velocity (Dates Unspecified)</b>", font=dict(size=14, color="#0F172A")),
            xaxis=dict(title="Units Sold", gridcolor="#E2E8F0"),
            yaxis=dict(title=None),
            margin=dict(l=10, r=10, t=40, b=20),
            height=320,
        )
        return fig

    return None


def create_expiry_risk_distribution_chart(df: pd.DataFrame):
    """Generates shelf-life expiry risk distribution bar chart using Plotly."""
    if df is None or df.empty:
        return None

    import plotly.graph_objects as go
    from datetime import datetime

    days_col = None
    if "days_to_expiry" in df.columns:
        days_col = pd.to_numeric(df["days_to_expiry"], errors="coerce")
    elif "expiry_date" in df.columns:
        parsed = pd.to_datetime(df["expiry_date"], errors="coerce").dt.date
        today = datetime.today().date()
        days_col = parsed.apply(lambda d: (d - today).days if pd.notnull(d) else np.nan)

    if days_col is None or days_col.dropna().empty:
        return None

    buckets = {
        "Expired (<0d)": int((days_col < 0).sum()),
        "Urgent (0-3d)": int(((days_col >= 0) & (days_col <= 3)).sum()),
        "Soon (4-7d)": int(((days_col > 3) & (days_col <= 7)).sum()),
        "Moderate (8-14d)": int(((days_col > 7) & (days_col <= 14)).sum()),
        "Safe (>14d)": int((days_col > 14).sum()),
    }

    colors = ["#DC2626", "#EA580C", "#F59E0B", "#84CC16", "#16A34A"]

    fig = go.Figure(
        go.Bar(
            x=list(buckets.keys()),
            y=list(buckets.values()),
            marker=dict(color=colors),
            text=list(buckets.values()),
            textposition="auto",
            hovertemplate="<b>%{x}</b><br>Product Count: %{y} SKUs<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=dict(text="<b>Expiry-Risk Distribution by Shelf-Life</b>", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Remaining Shelf-Life Horizon", gridcolor="#E2E8F0"),
        yaxis=dict(title="Number of SKUs", gridcolor="#E2E8F0"),
        margin=dict(l=10, r=10, t=40, b=20),
        height=320,
    )
    return fig


def categorize_inventory_dataframe(
    df: pd.DataFrame,
    low_stock_threshold: int = 10,
    expiry_alert_days: int = 5,
    excess_cover_days: int = 30,
) -> pd.DataFrame:
    """
    Categorizes products as Low Stock, Excess Stock, Nearing Expiry, Expired, or Normal
    based on actual inventory and sales data with configurable thresholds.
    Highlights urgent cases with unambiguous text labels and handles missing fields honestly.
    """
    from datetime import datetime, timedelta
    import numpy as np

    if df is None or df.empty:
        return pd.DataFrame()

    res = df.copy()
    today = datetime.today().date()

    # 1. Parse and resolve expiry date & days_to_expiry
    # Do NOT assume all grocery products have expiry dates!
    if "expiry_date" in res.columns:
        parsed_dates = pd.to_datetime(res["expiry_date"], errors="coerce")
        res["_parsed_expiry"] = parsed_dates.dt.date
        res["has_expiry_date"] = res["_parsed_expiry"].notna()
        res["days_to_expiry"] = res["_parsed_expiry"].apply(
            lambda d: (d - today).days if pd.notnull(d) else np.nan
        )
    elif "days_to_expiry" in res.columns:
        res["days_to_expiry"] = pd.to_numeric(res["days_to_expiry"], errors="coerce")
        res["has_expiry_date"] = res["days_to_expiry"].notna()
        res["_parsed_expiry"] = res["days_to_expiry"].apply(
            lambda d: (today + timedelta(days=int(d))) if pd.notnull(d) else None
        )
    else:
        res["has_expiry_date"] = False
        res["days_to_expiry"] = np.nan
        res["_parsed_expiry"] = None

    # 2. Resolve sales velocity for excess stock identification
    if "daily_sales_avg" in res.columns:
        res["_sales_velocity"] = pd.to_numeric(res["daily_sales_avg"], errors="coerce")
    elif "qty_sold" in res.columns:
        res["_sales_velocity"] = pd.to_numeric(res["qty_sold"], errors="coerce") / 7.0
    else:
        res["_sales_velocity"] = np.nan

    # 3. Resolve stock quantity & stock cover
    if "stock_qty" in res.columns:
        res["_numeric_stock"] = pd.to_numeric(res["stock_qty"], errors="coerce")
    else:
        res["_numeric_stock"] = np.nan

    # Stock cover in days = stock_qty / daily sales velocity
    res["stock_cover_days"] = np.where(
        (res["_sales_velocity"].notna()) & (res["_sales_velocity"] > 0) & (res["_numeric_stock"].notna()),
        (res["_numeric_stock"] / res["_sales_velocity"]).round(1),
        np.nan,
    )

    # 4. Resolve low stock threshold per item (use custom reorder_point column if available)
    if "reorder_point" in res.columns:
        res["_effective_reorder_pt"] = pd.to_numeric(res["reorder_point"], errors="coerce").fillna(low_stock_threshold)
    else:
        res["_effective_reorder_pt"] = float(low_stock_threshold)

    # 5. Row-by-row categorization with explicit labels
    categories = []
    status_labels = []
    urgency_ranks = []
    expiry_display_list = []
    sales_display_list = []
    stock_display_list = []

    for _, row in res.iterrows():
        stock = row["_numeric_stock"]
        unit_str = str(row.get("unit", "units")) if pd.notnull(row.get("unit")) else "units"
        has_expiry = bool(row["has_expiry_date"])
        days = row["days_to_expiry"]
        cover = row["stock_cover_days"]
        reorder_pt = row["_effective_reorder_pt"]
        velocity = row["_sales_velocity"]

        # Stock display
        if pd.notnull(stock):
            stock_display_list.append(f"{int(stock):,} {unit_str}")
        else:
            stock_display_list.append("— (Not Tracked)")

        # Expiry display
        if has_expiry and pd.notnull(days):
            exp_date_str = str(row.get("expiry_date", row["_parsed_expiry"]))
            days_int = int(days)
            if days_int < 0:
                expiry_display_list.append(f"{exp_date_str} (Expired {abs(days_int)}d ago)")
            elif days_int == 0:
                expiry_display_list.append(f"{exp_date_str} (Expires Today!)")
            else:
                expiry_display_list.append(f"{exp_date_str} ({days_int}d left)")
        else:
            expiry_display_list.append("— (No Expiry Date)")

        # Sales display
        if pd.notnull(velocity) and velocity > 0:
            if "qty_sold" in row and pd.notnull(row["qty_sold"]):
                sales_display_list.append(f"{int(row['qty_sold'])} sold ({velocity:.1f}/day)")
            else:
                sales_display_list.append(f"{velocity:.1f} units/day")
        else:
            sales_display_list.append("— (Not Tracked)")

        # Determine Category & Status with clear explicit text labels
        # Priority hierarchy:
        # 1. Expired
        # 2. Nearing Expiry
        # 3. Low Stock / Out of Stock
        # 4. Excess Stock
        # 5. Normal
        if has_expiry and pd.notnull(days) and days < 0:
            cat = "Expired"
            lbl = f"🚨 [CRITICAL: EXPIRED] Past Shelf-Life by {abs(int(days))}d"
            rank = 1
        elif has_expiry and pd.notnull(days) and 0 <= days <= expiry_alert_days:
            cat = "Nearing Expiry"
            lbl = f"⚠️ [URGENT: NEARING EXPIRY] {int(days)}d Remaining" if days > 0 else "⚠️ [URGENT: NEARING EXPIRY] Expires Today!"
            rank = 2
        elif pd.notnull(stock) and stock == 0:
            cat = "Low Stock"
            lbl = "📦 [ALERT: OUT OF STOCK] 0 Units Left"
            rank = 3
        elif pd.notnull(stock) and stock <= reorder_pt:
            cat = "Low Stock"
            lbl = f"🟡 [WARNING: LOW STOCK] {int(stock)} Units (Threshold: {int(reorder_pt)})"
            rank = 4
        elif pd.notnull(cover) and cover > excess_cover_days and pd.notnull(stock) and stock > reorder_pt:
            cat = "Excess Stock"
            lbl = f"📈 [NOTICE: EXCESS STOCK] {cover:.0f}d Cover (Limit: {excess_cover_days}d)"
            rank = 5
        elif pd.notnull(stock):
            cat = "Normal"
            if not has_expiry:
                lbl = "🟢 [NORMAL: HEALTHY STOCK] Non-Perishable"
            else:
                lbl = "🟢 [NORMAL: HEALTHY STOCK] Safe Shelf-Life"
            rank = 6
        else:
            cat = "Normal"
            lbl = "⚪ [NORMAL: STOCK UNSPECIFIED]"
            rank = 7

        categories.append(cat)
        status_labels.append(lbl)
        urgency_ranks.append(rank)

    res["stock_status"] = categories
    res["status_label"] = status_labels
    res["urgency_rank"] = urgency_ranks
    res["stock_display"] = stock_display_list
    res["expiry_display"] = expiry_display_list
    res["sales_display"] = sales_display_list

    res = res.drop(columns=["_parsed_expiry", "_sales_velocity", "_numeric_stock", "_effective_reorder_pt"], errors="ignore")
    return res
