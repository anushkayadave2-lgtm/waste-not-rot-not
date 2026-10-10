import sys
from pathlib import Path

# Ensure project root is in sys.path for Streamlit Cloud deployment
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
from modules.layout_helper import apply_custom_theme, setup_sidebar
from modules.data_processor import (
    generate_sample_data,
    calculate_dashboard_metrics,
    get_dashboard_alerts,
    create_category_inventory_chart,
    create_sales_trend_chart,
    create_expiry_risk_distribution_chart,
)

# ---------------------------------------------------------
# Page Config & Shared Theme Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard | Zero-Waste Grocery",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_theme()
setup_sidebar()

# ---------------------------------------------------------
# Session State Initialization (No hardcoded demo by default)
# ---------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "data_uploaded" not in st.session_state:
    st.session_state.data_uploaded = False

if "is_demo" not in st.session_state:
    st.session_state.is_demo = False

df_raw = st.session_state.get("df")

# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown("## 🍃 Grocery Operations Dashboard")
st.caption("Live store inventory tracking, shelf-life expiry alerts, and historical sales performance.")

# ---------------------------------------------------------
# Empty State: Friendly Guide When No Data Has Been Uploaded
# ---------------------------------------------------------
if df_raw is None or getattr(df_raw, "empty", True):
    st.markdown(
        """
        <div class="section-card-green" style="padding: 2rem; border-radius: 14px; margin-top: 1rem;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">👋 Welcome to Zero-Waste Grocery!</div>
            <h3 style="color: #15803D; margin-top: 0; font-weight: 700;">No Store Dataset Loaded Yet</h3>
            <p style="color: #166534; font-size: 1.05rem; line-height: 1.6; max-width: 820px; margin-bottom: 1.25rem;">
                This dashboard dynamically calculates all store metrics directly from your uploaded store inventory
                or sales spreadsheets. To get started, please import your store data or explore using sample records.
            </p>
            <div style="background-color: #FFFFFF; border: 1px solid #BBF7D0; border-radius: 10px; padding: 1.25rem; margin-bottom: 1.5rem;">
                <h4 style="margin-top:0; color: #166534;">🚀 How it works in 3 easy steps:</h4>
                <ol style="margin-bottom:0; color: #334155; line-height: 1.8;">
                    <li><strong>Upload File:</strong> Navigate to <em>Upload Data</em> and import any CSV or Excel (.xlsx) spreadsheet.</li>
                    <li><strong>Auto-Map Columns:</strong> Confirm column mappings for product names, stock quantities, expiry dates, or sales.</li>
                    <li><strong>Live Dashboard:</strong> Monitor real-time low-stock alerts, shelf-life risk distributions, and store metrics.</li>
                </ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, col_spacer = st.columns([1.5, 1.5, 3])

    with col_btn1:
        if st.button("📤 Upload Store Spreadsheet", use_container_width=True):
            st.switch_page("pages/1_📤_Upload_Data.py")

    with col_btn2:
        if st.button("✨ Load Sample Demo Data", use_container_width=True):
            st.session_state.df = generate_sample_data()
            st.session_state.data_uploaded = True
            st.session_state.is_demo = True
            st.session_state.uploaded_filename = "Built-in Sample Grocery Dataset"
            st.rerun()

    st.markdown("---")
    st.caption("Zero-Waste Grocery Management System | Non-Technical Store Owner Interface")
    st.stop()

# ---------------------------------------------------------
# Active Dataset Status Banner
# ---------------------------------------------------------
df = df_raw.copy()
is_demo = st.session_state.get("is_demo", False)
uploaded_file_name = st.session_state.get("uploaded_filename", "Uploaded Spreadsheet")

if is_demo:
    st.markdown(
        f"""
        <div class="section-card" style="border-left: 5px solid #F59E0B; padding: 0.9rem 1.25rem; margin-bottom: 1.25rem; background-color: #FFFBEB;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong style="color: #B45309;">🟡 Demo Mode Active:</strong>
                    <span style="color: #92400E; font-size: 0.95rem;">
                        You are viewing built-in sample grocery inventory data (25 items). 
                        To view your actual store metrics, upload your store spreadsheet on the <strong>Upload Data</strong> page.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="section-card-green" style="padding: 0.85rem 1.25rem; margin-bottom: 1.25rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong style="color: #15803D;">🟢 Live Store Data Active:</strong>
                    <span style="color: #166534; font-size: 0.95rem;">
                        Displaying metrics calculated directly from <code>{uploaded_file_name}</code> ({len(df):,} total records).
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Simple Dashboard Filters (Category, Expiry Horizon, Sales Dates)
# ---------------------------------------------------------
st.markdown("### 🔍 Dashboard Filters")
st.caption("Filter inventory and sales data by category, remaining shelf-life, or date span.")

col_f1, col_f2, col_f3, col_f4 = st.columns(4)

# 1. Category Filter
with col_f1:
    if "category" in df.columns:
        all_cats = ["All Categories"] + sorted([str(c) for c in df["category"].dropna().unique() if str(c).strip()])
        selected_category = st.selectbox("Product Category", options=all_cats, index=0)
    else:
        st.selectbox("Product Category", options=["All Categories (Column Unavailable)"], disabled=True)
        selected_category = "All Categories"

# 2. Expiry Horizon Filter
with col_f2:
    if "days_to_expiry" in df.columns or "expiry_date" in df.columns:
        expiry_filter_options = [
            "All Shelf-Life Horizons",
            "🚨 Already Expired (< 0 Days)",
            "⚠️ Urgent Expiry (≤ 3 Days)",
            "🌱 Expiring Soon (≤ 7 Days)",
            "📅 Expiring This Month (≤ 30 Days)",
        ]
        selected_expiry_horizon = st.selectbox("Expiry Date Horizon", options=expiry_filter_options, index=0)
    else:
        st.selectbox("Expiry Date Horizon", options=["All Horizons (Expiry Unavailable)"], disabled=True)
        selected_expiry_horizon = "All Shelf-Life Horizons"

# 3. Sales Date Filter (when sale_date exists)
with col_f3:
    if "sale_date" in df.columns and not df["sale_date"].dropna().empty:
        valid_sale_dates = pd.to_datetime(df["sale_date"], errors="coerce").dropna().dt.date
        if not valid_sale_dates.empty:
            min_date = valid_sale_dates.min()
            max_date = valid_sale_dates.max()
            date_range = st.date_input(
                "Sales Date Window",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
        else:
            date_range = None
            st.text_input("Sales Date Window", value="No valid dates found", disabled=True)
    else:
        st.text_input("Sales Date Window", value="Sales date column not mapped", disabled=True)
        date_range = None

# 4. Low-stock Threshold Adjustment
with col_f4:
    has_reorder_col = "reorder_point" in df.columns
    if has_reorder_col:
        low_stock_threshold = st.number_input(
            "Low Stock Threshold (Fallback)",
            min_value=1,
            max_value=100,
            value=10,
            help="Your dataset has custom 'reorder_point' values which take priority; this number serves as fallback for unassigned items.",
        )
    elif "stock_qty" in df.columns:
        low_stock_threshold = st.number_input(
            "Low Stock Threshold (Units)",
            min_value=1,
            max_value=100,
            value=10,
            help="Products with stock at or below this threshold are flagged as low stock.",
        )
    else:
        st.number_input("Low Stock Threshold", min_value=1, max_value=100, value=10, disabled=True)
        low_stock_threshold = 10

# Apply Category & Expiry Filtering
filtered_df = df.copy()

if selected_category != "All Categories" and "category" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["category"] == selected_category]

# Ensure days_to_expiry column is populated if expiry_date exists
if "days_to_expiry" not in filtered_df.columns and "expiry_date" in filtered_df.columns:
    parsed_dates = pd.to_datetime(filtered_df["expiry_date"], errors="coerce").dt.date
    today_dt = datetime.today().date()
    filtered_df["days_to_expiry"] = parsed_dates.apply(lambda d: (d - today_dt).days if pd.notnull(d) else None)

if "days_to_expiry" in filtered_df.columns:
    if selected_expiry_horizon == "🚨 Already Expired (< 0 Days)":
        filtered_df = filtered_df[filtered_df["days_to_expiry"] < 0]
    elif selected_expiry_horizon == "⚠️ Urgent Expiry (≤ 3 Days)":
        filtered_df = filtered_df[filtered_df["days_to_expiry"] <= 3]
    elif selected_expiry_horizon == "🌱 Expiring Soon (≤ 7 Days)":
        filtered_df = filtered_df[filtered_df["days_to_expiry"] <= 7]
    elif selected_expiry_horizon == "📅 Expiring This Month (≤ 30 Days)":
        filtered_df = filtered_df[filtered_df["days_to_expiry"] <= 30]

# Calculate Dashboard Metrics from Filtered Dataset
metrics = calculate_dashboard_metrics(filtered_df, low_stock_threshold=low_stock_threshold)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Metric Cards Section
# Clearly distinguishes unavailable metrics from zero values
# ---------------------------------------------------------
st.markdown("### 📊 Key Inventory & Operations Metrics")

m1, m2, m3, m4, m5 = st.columns(5)

# Card 1: Distinct Products
with m1:
    if metrics["distinct_products_available"]:
        val_str = f"{metrics['distinct_products']:,}"
        sub_str = "Distinct SKUs / Items"
        box_class = "metric-card-box"
        val_class = "metric-card-val"
    else:
        val_str = "N/A"
        sub_str = "Identifier not mapped"
        box_class = "metric-card-box-na"
        val_class = "metric-card-val-na"

    st.markdown(
        f"""
        <div class="{box_class}">
            <div class="metric-card-lbl">Distinct Products</div>
            <div class="{val_class}">{val_str}</div>
            <div class="metric-card-sub">{sub_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Card 2: Total Stock Quantity (with Unit Compatibility check)
with m2:
    if metrics["stock_qty_available"]:
        if metrics["units_compatible"]:
            val_str = f"{int(metrics['total_stock_qty']):,}"
            sub_str = f"Compatible unit: <strong>{metrics['unit_label']}</strong>"
            box_class = "metric-card-box"
            val_class = "metric-card-val"
        else:
            val_str = "Mixed Units"
            sub_str = "Incompatible units detected"
            box_class = "metric-card-box"
            val_class = "metric-card-val"
    else:
        val_str = "N/A"
        sub_str = "stock_qty not mapped"
        box_class = "metric-card-box-na"
        val_class = "metric-card-val-na"

    st.markdown(
        f"""
        <div class="{box_class}">
            <div class="metric-card-lbl">Total Stock Quantity</div>
            <div class="{val_class}">{val_str}</div>
            <div class="metric-card-sub">{sub_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Card 3: Products with Low Stock
with m3:
    if metrics["low_stock_available"]:
        low_cnt = metrics["low_stock_count"]
        val_str = f"{low_cnt:,} SKUs"
        sub_str = f"Threshold: {metrics['low_stock_threshold_label']}"
        box_class = "metric-card-box"
        val_class = "metric-card-val-alert" if low_cnt > 0 else "metric-card-val"
    else:
        val_str = "N/A"
        sub_str = "stock_qty not mapped"
        box_class = "metric-card-box-na"
        val_class = "metric-card-val-na"

    st.markdown(
        f"""
        <div class="{box_class}">
            <div class="metric-card-lbl">Low Stock Products</div>
            <div class="{val_class}">{val_str}</div>
            <div class="metric-card-sub">{sub_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Card 4: Products Nearing Expiry or Already Expired
with m4:
    if metrics["expiry_available"]:
        exp_cnt = metrics["expired_count"]
        urg_cnt = metrics["expiring_soon_3d"]
        total_risk = exp_cnt + urg_cnt
        val_str = f"{total_risk:,} SKUs"
        sub_str = f"{exp_cnt} expired • {urg_cnt} expiring ≤ 3d"
        box_class = "metric-card-box"
        val_class = "metric-card-val-alert" if total_risk > 0 else "metric-card-val"
    else:
        val_str = "N/A"
        sub_str = "expiry_date not mapped"
        box_class = "metric-card-box-na"
        val_class = "metric-card-val-na"

    st.markdown(
        f"""
        <div class="{box_class}">
            <div class="metric-card-lbl">Expiry Risk (≤ 3d)</div>
            <div class="{val_class}">{val_str}</div>
            <div class="metric-card-sub">{sub_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Card 5: Inventory Valuation / Sales Summary Link
with m5:
    if metrics["valuation_available"]:
        val_str = f"${metrics['total_valuation']:,.2f}"
        sub_str = "Valuation (Stock × Price)"
        box_class = "metric-card-box"
        val_class = "metric-card-val"
    elif metrics["sales_available"] and metrics["total_qty_sold"] is not None:
        val_str = f"{int(metrics['total_qty_sold']):,} units"
        sub_str = f"Velocity: {metrics['avg_daily_sales']}/day"
        box_class = "metric-card-box"
        val_class = "metric-card-val"
    else:
        val_str = "N/A"
        sub_str = "Price & Sales not mapped"
        box_class = "metric-card-box-na"
        val_class = "metric-card-val-na"

    st.markdown(
        f"""
        <div class="{box_class}">
            <div class="metric-card-lbl">Inventory Valuation</div>
            <div class="{val_class}">{val_str}</div>
            <div class="metric-card-sub">{sub_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Detail Expander for Mixed Units when applicable
if metrics["stock_qty_available"] and not metrics["units_compatible"] and metrics["unit_breakdown"]:
    with st.expander("ℹ️ View Incompatible Units Breakdown"):
        st.write("Quantities in your spreadsheet use multiple incompatible units of measure:")
        unit_cols = st.columns(len(metrics["unit_breakdown"]))
        for idx, (u_name, u_qty) in enumerate(metrics["unit_breakdown"].items()):
            with unit_cols[idx]:
                st.metric(label=f"Unit: {u_name.upper()}", value=f"{u_qty:,.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Short List of Most Important Stock & Expiry Alerts
# ---------------------------------------------------------
st.markdown("### ⚡ Critical Stock & Expiry Alerts")
st.caption("Actionable priority feed for expired products, immediate shelf-life risks, and stockouts.")

alerts = get_dashboard_alerts(filtered_df, low_stock_threshold=low_stock_threshold, max_alerts=6)

if alerts:
    for alt in alerts:
        st.markdown(
            f"""
            <div class="alert-card-item" style="border-left: 5px solid {alt['border_color']};">
                <div style="flex-grow: 1;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                        <span class="status-pill" style="background-color: {alt['badge_bg']}; color: {alt['badge_color']};">
                            {alt['badge']}
                        </span>
                        <strong style="color: #0F172A; font-size: 1rem;">{alt['item_name']}</strong>
                        <span style="color: #64748B; font-size: 0.85rem;">({alt['category']})</span>
                    </div>
                    <div style="color: #475569; font-size: 0.9rem;">{alt['message']}</div>
                </div>
                <div style="text-align: right; min-width: 220px;">
                    <span style="background-color: #F8FAFC; border: 1px solid #CBD5E1; color: #334155; font-size: 0.8rem; font-weight: 600; padding: 4px 10px; border-radius: 6px;">
                        👉 Action: {alt['action']}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    if metrics["stock_qty_available"] or metrics["expiry_available"]:
        st.markdown(
            """
            <div class="section-card-green" style="padding: 1rem 1.25rem;">
                <strong style="color: #15803D;">✅ Inventory Healthy:</strong>
                <span style="color: #166534;">
                    No urgent expiry or low-stock alerts detected for the selected filter view.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="section-card" style="padding: 1rem 1.25rem;">
                <strong style="color: #64748B;">ℹ️ Alerts Unavailable:</strong>
                <span style="color: #64748B;">
                    Stock quantity and expiry date columns are not present in this dataset to generate alerts.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Charts Section
# 1. Category-wise inventory
# 2. Expiry-risk distribution
# 3. Sales trends
# ---------------------------------------------------------
st.markdown("### 📈 Visual Distribution & Analytics")
st.caption("Visual breakdowns of store stock, shelf-life vulnerability, and sales velocity.")

c_col1, c_col2 = st.columns(2)

# Chart 1: Category-Wise Inventory
with c_col1:
    fig_cat = create_category_inventory_chart(filtered_df)
    if fig_cat is not None:
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="section-card" style="height: 320px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📦</div>
                <h4 style="margin: 0; color: #475569;">Category Inventory Chart Unavailable</h4>
                <p style="color: #94A3B8; font-size: 0.9rem; max-width: 320px; margin-top: 0.5rem;">
                    The current dataset does not include a mapped <code>category</code> column.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Chart 2: Expiry-Risk Distribution
with c_col2:
    fig_expiry = create_expiry_risk_distribution_chart(filtered_df)
    if fig_expiry is not None:
        st.plotly_chart(fig_expiry, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="section-card" style="height: 320px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🗓️</div>
                <h4 style="margin: 0; color: #475569;">Expiry-Risk Distribution Unavailable</h4>
                <p style="color: #94A3B8; font-size: 0.9rem; max-width: 320px; margin-top: 0.5rem;">
                    The current dataset does not include a mapped <code>expiry_date</code> column.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Chart 3: Historical Sales Trend
st.markdown("#### 🛒 Historical Sales Performance")

start_date_filter = None
end_date_filter = None
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date_filter = date_range[0]
    end_date_filter = date_range[1]

fig_sales = create_sales_trend_chart(filtered_df, start_date=start_date_filter, end_date=end_date_filter)

if fig_sales is not None:
    st.plotly_chart(fig_sales, use_container_width=True)
else:
    st.markdown(
        """
        <div class="section-card" style="padding: 1.5rem; text-align: center;">
            <strong style="color: #475569;">📈 Sales Trends Chart Unavailable</strong>
            <p style="color: #94A3B8; margin-top: 0.25rem; margin-bottom: 0;">
                No historical sales records (<code>sale_date</code> or <code>qty_sold</code>) are available in this dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Historical Sales Summary Section
# ---------------------------------------------------------
st.markdown("### 📋 Historical Sales Summary")

if metrics["sales_available"]:
    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.markdown(
            f"""
            <div class="section-card" style="padding: 1rem; text-align: center;">
                <div class="metric-card-lbl">Total Units Sold</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #15803D;">
                    {int(metrics['total_qty_sold']) if metrics['total_qty_sold'] is not None else 0:,}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s2:
        rev_str = f"${metrics['total_sales_revenue']:,.2f}" if metrics["total_sales_revenue"] is not None else "Price N/A"
        st.markdown(
            f"""
            <div class="section-card" style="padding: 1rem; text-align: center;">
                <div class="metric-card-lbl">Total Sales Revenue</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #15803D;">
                    {rev_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s3:
        st.markdown(
            f"""
            <div class="section-card" style="padding: 1rem; text-align: center;">
                <div class="metric-card-lbl">Average Daily Velocity</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #15803D;">
                    {metrics['avg_daily_sales']} units/day
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with s4:
        if metrics["sales_date_min"] and metrics["sales_date_max"]:
            window_str = f"{metrics['sales_date_min']} to {metrics['sales_date_max']}"
        else:
            window_str = "Dates Unspecified"

        st.markdown(
            f"""
            <div class="section-card" style="padding: 1rem; text-align: center;">
                <div class="metric-card-lbl">Sales Time Horizon</div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #475569; margin-top: 6px;">
                    {window_str}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        """
        <div class="section-card" style="padding: 1.25rem;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.4rem;">📊</span>
                <div>
                    <strong style="color: #475569;">Sales Summary Unavailable:</strong>
                    <span style="color: #64748B;">
                        This store upload does not contain sales velocity or transaction columns (<code>qty_sold</code>, <code>sale_date</code>).
                        Upload sales data via <strong>Upload Data</strong> to unlock revenue metrics and velocity summaries.
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Filtered Inventory Table Preview
# ---------------------------------------------------------
with st.expander(f"📦 View Filtered Product Records ({len(filtered_df)} items)", expanded=False):
    st.dataframe(filtered_df, use_container_width=True, height=350)

st.markdown("---")
st.caption("Zero-Waste Grocery Operations Dashboard | Built with dynamic metrics from upload and processing modules.")
