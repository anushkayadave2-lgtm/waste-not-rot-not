from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Standard internal schema definition with common column aliases
STANDARD_FIELDS = {
    "item_id": {
        "label": "Item ID / SKU",
        "aliases": ["item_id", "sku", "product_id", "id", "code", "sku_code", "item_code", "barcode"],
        "required_for": ["Item Identification"],
    },
    "item_name": {
        "label": "Product Name",
        "aliases": ["product_name", "item_name", "product", "item", "title", "name", "product_title", "description"],
        "required_for": ["Inventory Tracking", "Recommendations"],
    },
    "category": {
        "label": "Category",
        "aliases": ["category", "department", "section", "group", "item_category", "prod_category", "type"],
        "required_for": ["Category Breakdown"],
    },
    "stock_qty": {
        "label": "Quantity in Stock",
        "aliases": ["stock_qty", "quantity_in_stock", "stock", "quantity", "qty", "inventory", "in_stock", "current_stock", "units"],
        "required_for": ["Inventory Tracking", "Reorder Alerts", "Stockout Risk"],
    },
    "unit_price": {
        "label": "Unit Price",
        "aliases": ["unit_price", "price", "cost", "selling_price", "retail_price", "rate", "unit_cost"],
        "required_for": ["Stock Valuation", "Revenue Recovery", "Financial Reports"],
    },
    "expiry_date": {
        "label": "Expiry Date",
        "aliases": ["expiry_date", "expiration_date", "expiry", "best_before", "use_by", "exp_date", "expiration", "sell_by"],
        "required_for": ["Expiry Tracking", "Waste Risk Score", "Dynamic Markdowns", "Donation Routing"],
    },
    "qty_sold": {
        "label": "Quantity Sold / Sales Velocity",
        "aliases": ["qty_sold", "quantity_sold", "sales_qty", "units_sold", "daily_sales", "sales", "daily_sales_avg", "velocity"],
        "required_for": ["Demand Forecasting", "Reorder Leveling"],
    },
    "sale_date": {
        "label": "Sale / Transaction Date",
        "aliases": ["sale_date", "date_sold", "transaction_date", "date", "sales_date", "order_date"],
        "required_for": ["Sales Trend Analysis"],
    },
    "discount": {
        "label": "Discount / Markdown",
        "aliases": ["discount", "markdown", "promo", "discount_pct", "discount_amount"],
        "required_for": ["Promotional Analysis"],
    },
    "reorder_point": {
        "label": "Reorder Threshold / Point",
        "aliases": ["reorder_point", "reorder_level", "min_stock", "safety_stock", "reorder_threshold", "reorder_qty", "threshold"],
        "required_for": ["Low Stock Alerts"],
    },
    "unit": {
        "label": "Unit of Measurement",
        "aliases": ["unit", "uom", "unit_of_measure", "measurement_unit", "stock_unit", "qty_unit"],
        "required_for": ["Unit Compatibility Check"],
    },
}


def auto_map_columns(raw_columns: List[str]) -> Dict[str, str]:
    """
    Attempts to automatically map incoming raw CSV/XLSX column headers
    to standard internal field keys based on common aliases.
    Returns dict: {raw_column_name: standard_field_key}
    """
    col_mapping = {}
    used_standards = set()

    # Clean raw column strings
    raw_col_clean = {c: str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in raw_columns}

    for raw_original, raw_clean in raw_col_clean.items():
        matched = False
        for std_key, meta in STANDARD_FIELDS.items():
            if std_key in used_standards:
                continue
            for alias in meta["aliases"]:
                alias_clean = alias.strip().lower().replace(" ", "_").replace("-", "_")
                if raw_clean == alias_clean or alias_clean in raw_clean:
                    col_mapping[raw_original] = std_key
                    used_standards.add(std_key)
                    matched = True
                    break
            if matched:
                break

    return col_mapping


def normalize_dataframe(raw_df: pd.DataFrame, col_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Converts raw DataFrame into a standardized internal format using col_mapping.
    Performs data cleaning: type coercion, duplicate removal, invalid date parsing,
    non-negative quantity bounds. Does NOT invent missing values.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), {"status": "Error", "message": "Empty DataFrame provided."}

    df = raw_df.copy()

    # Filter mapping to only include existing columns
    valid_mapping = {k: v for k, v in col_mapping.items() if k in df.columns and v != "ignore"}

    # Rename mapped columns
    df = df.rename(columns=valid_mapping)

    # Record cleaning audit report
    initial_rows = len(df)
    cleaning_report = {
        "initial_rows": initial_rows,
        "duplicates_removed": 0,
        "invalid_dates_coerced": 0,
        "negative_quantities_fixed": 0,
        "mapped_fields": list(valid_mapping.values()),
        "unmapped_raw_columns": [c for c in raw_df.columns if c not in valid_mapping],
    }

    # 1. Deduplication
    df_dedup = df.drop_duplicates()
    cleaning_report["duplicates_removed"] = initial_rows - len(df_dedup)
    df = df_dedup

    # 2. Clean Numeric Fields
    numeric_fields = ["stock_qty", "unit_price", "qty_sold", "discount", "reorder_point"]
    for num_col in numeric_fields:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            neg_count = (df[num_col] < 0).sum()
            if neg_count > 0:
                cleaning_report["negative_quantities_fixed"] += int(neg_count)
                df[num_col] = df[num_col].apply(lambda x: max(0.0, float(x)) if pd.notnull(x) else np.nan)

    # 3. Clean Expiry Date & Calculate days_to_expiry
    if "expiry_date" in df.columns:
        parsed_dates = pd.to_datetime(df["expiry_date"], errors="coerce")
        invalid_dates = parsed_dates.isna() & df["expiry_date"].notna()
        cleaning_report["invalid_dates_coerced"] = int(invalid_dates.sum())

        today = datetime.today().date()
        df["expiry_date"] = parsed_dates.dt.date
        df["days_to_expiry"] = df["expiry_date"].apply(lambda d: (d - today).days if pd.notnull(d) else np.nan)

    # 4. Clean Sale Date
    if "sale_date" in df.columns:
        df["sale_date"] = pd.to_datetime(df["sale_date"], errors="coerce").dt.date

    # 5. Clean String Fields
    string_fields = ["item_id", "item_name", "category", "unit"]
    for str_col in string_fields:
        if str_col in df.columns:
            df[str_col] = df[str_col].astype(str).str.strip()
            df[str_col] = df[str_col].replace(["nan", "None", "NaN", ""], np.nan)

    # Fill daily_sales_avg if qty_sold is available
    if "daily_sales_avg" not in df.columns and "qty_sold" in df.columns:
        df["daily_sales_avg"] = df["qty_sold"]

    cleaning_report["final_rows"] = len(df)
    return df, cleaning_report


def check_feature_support(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Inspects normalized DataFrame and explains which platform features
    can operate vs. which features are disabled due to missing fields.
    """
    cols = set(df.columns) if df is not None else set()

    features = {
        "Inventory Tracking": {
            "required": ["item_name", "stock_qty"],
            "supported": {"item_name", "stock_qty"}.issubset(cols),
            "missing": list({"item_name", "stock_qty"} - cols),
            "description": "Displays SKU stock balances, category totals, and unit counts.",
        },
        "Expiry & Waste Tracking": {
            "required": ["expiry_date"],
            "supported": "expiry_date" in cols,
            "missing": list({"expiry_date"} - cols),
            "description": "Flags products expiring within 3 to 7 days and calculates waste vulnerability.",
        },
        "Stock Valuation & Financials": {
            "required": ["unit_price", "stock_qty"],
            "supported": {"unit_price", "stock_qty"}.issubset(cols),
            "missing": list({"unit_price", "stock_qty"} - cols),
            "description": "Calculates total store inventory valuation and dollar value at risk of waste.",
        },
        "Demand Forecasting": {
            "required": ["qty_sold", "stock_qty"],
            "supported": (("qty_sold" in cols or "daily_sales_avg" in cols) and "stock_qty" in cols),
            "missing": [c for c in ["qty_sold / daily_sales_avg", "stock_qty"] if c not in cols and "daily_sales_avg" not in cols],
            "description": "Uses historical sales velocity to project future item demand and stockout risks.",
        },
        "Dynamic Markdowns & Donations": {
            "required": ["expiry_date", "unit_price"],
            "supported": {"expiry_date", "unit_price"}.issubset(cols),
            "missing": list({"expiry_date", "unit_price"} - cols),
            "description": "Recommends time-decay discounts (50%, 30%, 15%) and packages food bank donations.",
        },
    }

    return features
