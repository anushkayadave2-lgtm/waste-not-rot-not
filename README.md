# 🍃 Zero-Waste Grocery

An intelligent, multi-agent grocery operations platform designed to help store managers reduce food waste, optimize replenishment, minimize shelf-life markdowns, and prevent stockouts.

Built with **Streamlit**, **Pandas**, **Scikit-Learn**, and **Plotly**.

---

## 🌟 Key Features

1. **📊 Live Operations Dashboard (`app.py`)**
   - High-level KPIs: distinct SKUs, total stock valuation, low-stock warnings, and expiry alerts.
   - Interactive Plotly visualizations for category distribution, sales trends, and shelf-life urgency.
   - Friendly empty state with one-click demo data loading.

2. **📤 Store Data Import & Intelligent Mapping (`pages/1_📤_Upload_Data.py`)**
   - Upload any CSV or Excel (`.xlsx`) inventory or sales spreadsheet.
   - Intelligent fuzzy column mapper to automatically match custom store headers to standardized internal schemas.
   - Data cleaning engine: deduplication, date normalization, negative quantity coercion, and downloadable templates.

3. **📦 Inventory & Shelf-Life Tracker (`pages/2_📦_Inventory_&_Expiry.py`)**
   - Multi-parameter filtering by stock status (Low Stock, Excess Stock, Nearing Expiry, Expired, Normal).
   - Configurable low-stock thresholds and expiry alert timeframes.
   - One-click filtered CSV inventory exports.

4. **📈 Store Demand Forecasting (`pages/3_📈_Demand_Forecast.py`)**
   - Daily sales aggregation from historical transactions.
   - Moving average baseline vs. Scikit-Learn regression models (Ridge, Decision Tree, Random Forest).
   - Out-of-sample backtest evaluation reporting WAPE, MAE, and RMSE metrics.

5. **🤖 AI Recommendations & Decision Center (`pages/4_🤖_AI_Recommendations.py`)**
   - Five specialized deterministic agents coordinated by priority scoring:
     - **Demand Agent:** Identifies stockouts and demand velocity.
     - **Freshness Agent:** Highlights critical spoilage windows.
     - **Discount Agent:** Suggests graduated markdowns (15% / 30% / 50%) based on days to expiry.
     - **Replenishment Agent:** Recommends reorder quantities factoring in lead time and sales velocity.
     - **Transfer Agent:** Proposes inter-store stock transfers when multi-location records exist.
   - Human-in-the-loop decision workflow: Approve, Reject, or Modify with full session audit trail and CSV download.

6. **📑 Store Waste & Operations Audit Reports (`pages/5_📑_Reports.py`)**
   - Executive briefing with actionable summaries.
   - Comprehensive multi-sheet Excel workbook export (`.xlsx`) and consolidated CSV download.

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/zero-waste-grocery.git
   cd zero-waste-grocery
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Running Automated Tests

Run the included unit and integration test suite:
```bash
python test_app.py
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push your repository to **GitHub**.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your repository, set the branch to `main`, and specify the main file path as `app.py`.
5. Click **"Deploy!"**.
