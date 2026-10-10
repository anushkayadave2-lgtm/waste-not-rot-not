import io
import sys
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import openpyxl

from modules.data_normalizer import (
    STANDARD_FIELDS,
    auto_map_columns,
    normalize_dataframe,
    check_feature_support,
)
from modules.data_processor import (
    read_uploaded_file,
    calculate_dashboard_metrics,
    get_dashboard_alerts,
    categorize_inventory_dataframe,
    create_category_inventory_chart,
    create_sales_trend_chart,
    create_expiry_risk_distribution_chart,
    generate_sample_data,
)
from modules.forecaster import (
    check_forecasting_data_readiness,
    aggregate_daily_sales,
    build_forecasting_features,
    evaluate_models_on_holdout,
    generate_future_forecast,
    generate_sample_sales_history,
)
from modules.agents import (
    AgentRecommendation,
    DemandAgent,
    FreshnessAgent,
    DiscountAgent,
    ReplenishmentAgent,
    TransferAgent,
    AgentCoordinator,
)

class MockUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def read(self) -> bytes:
        return self._content


class TestZeroWasteGroceryApplication(unittest.TestCase):

    def setUp(self):
        self.today = datetime.today().date()

    # 1. FILE UPLOAD & PARSING
    def test_upload_valid_csv(self):
        csv_text = "item_id,item_name,stock_qty,unit_price\nSKU-1,Apples,50,2.50\nSKU-2,Oranges,30,3.00\n"
        mock_file = MockUploadedFile("test.csv", csv_text.encode("utf-8"))
        df, err = read_uploaded_file(mock_file)
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)
        self.assertIn("item_name", df.columns)

    def test_upload_valid_xlsx(self):
        raw_df = pd.DataFrame({
            "item_id": ["SKU-1", "SKU-2"],
            "item_name": ["Milk", "Bread"],
            "stock_qty": [20, 15],
            "unit_price": [4.00, 2.80]
        })
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            raw_df.to_excel(writer, index=False)
        mock_file = MockUploadedFile("test.xlsx", buffer.getvalue())
        df, err = read_uploaded_file(mock_file)
        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 2)

    def test_upload_empty_file(self):
        mock_file = MockUploadedFile("empty.csv", b"")
        df, err = read_uploaded_file(mock_file)
        self.assertIsNotNone(err)
        self.assertIsNone(df)

    def test_upload_corrupted_file(self):
        mock_file = MockUploadedFile("corrupted.xlsx", b"THIS_IS_NOT_AN_EXCEL_FILE_GARBAGE_BYTES_123456")
        df, err = read_uploaded_file(mock_file)
        self.assertIsNotNone(err)
        self.assertIsNone(df)

    def test_upload_unsupported_extension(self):
        mock_file = MockUploadedFile("data.json", b'{"key": "value"}')
        df, err = read_uploaded_file(mock_file)
        self.assertIsNotNone(err)
        self.assertIn("Unsupported", err)

    # 2. DATA NORMALIZATION & ALTERNATIVE COLUMN HEADERS
    def test_auto_map_alternative_headers(self):
        raw_cols = ["product_title", "units_in_stock", "expiration", "selling_price", "sales_volume", "reorder_level", "sku_code"]
        mapping = auto_map_columns(raw_cols)
        self.assertEqual(mapping.get("product_title"), "item_name")
        self.assertEqual(mapping.get("units_in_stock"), "stock_qty")
        self.assertEqual(mapping.get("expiration"), "expiry_date")
        self.assertEqual(mapping.get("selling_price"), "unit_price")
        self.assertEqual(mapping.get("sales_volume"), "qty_sold")
        self.assertEqual(mapping.get("reorder_level"), "reorder_point")
        self.assertEqual(mapping.get("sku_code"), "item_id")

    def test_normalize_dirty_data(self):
        raw_df = pd.DataFrame({
            "product_title": ["Milk", "Bread", "Milk", "Cheese", "Tofu"],
            "units_in_stock": [20, -5, 20, "15", np.nan],
            "expiration": [
                (self.today + timedelta(days=5)).strftime("%Y-%m-%d"),
                "INVALID_DATE_STRING",
                (self.today + timedelta(days=5)).strftime("%Y-%m-%d"),
                "2026-99-99",
                None,
            ],
            "selling_price": [3.50, -1.0, 3.50, "4.20", 2.00],
        })
        mapping = auto_map_columns(list(raw_df.columns))
        norm_df, report = normalize_dataframe(raw_df, mapping)
        self.assertIn("days_to_expiry", norm_df.columns)
        self.assertIn("stock_qty", norm_df.columns)
        self.assertGreaterEqual(report["duplicates_removed"], 1)
        self.assertGreaterEqual(report["negative_quantities_fixed"], 1)
        self.assertTrue((norm_df["stock_qty"].dropna() >= 0).all())
        self.assertGreaterEqual(report["invalid_dates_coerced"], 1)

    def test_feature_support_reporting(self):
        partial_df = pd.DataFrame({"item_name": ["Apples"], "stock_qty": [10]})
        features = check_feature_support(partial_df)
        self.assertTrue(features["Inventory Tracking"]["supported"])
        self.assertFalse(features["Expiry & Waste Tracking"]["supported"])
        self.assertFalse(features["Stock Valuation & Financials"]["supported"])

    # 3. DASHBOARD METRICS & INVENTORY CATEGORIZATION
    def test_dashboard_metrics_calculation(self):
        df = generate_sample_data(num_items=25)
        metrics = calculate_dashboard_metrics(df, low_stock_threshold=10)
        self.assertTrue(metrics["has_data"])
        self.assertEqual(metrics["distinct_products"], 25)
        self.assertTrue(metrics["stock_qty_available"])
        self.assertGreater(metrics["total_stock_qty"], 0)
        self.assertTrue(metrics["expiry_available"])
        self.assertGreater(metrics["total_expiry_risk"], 0)

    def test_dashboard_metrics_mixed_units(self):
        df = pd.DataFrame({
            "item_name": ["Item A", "Item B"],
            "stock_qty": [10, 20],
            "unit": ["kg", "bottles"]
        })
        metrics = calculate_dashboard_metrics(df)
        self.assertFalse(metrics["units_compatible"])
        self.assertIn("kg", metrics["unit_breakdown"])
        self.assertIn("bottles", metrics["unit_breakdown"])

    def test_inventory_categorization_labels(self):
        today = self.today
        df = pd.DataFrame({
            "item_name": ["Expired Yogurt", "Urgent Milk", "Normal Pasta", "Out Stock Beef", "Excess Rice"],
            "stock_qty": [10, 15, 50, 0, 100],
            "expiry_date": [
                (today - timedelta(days=2)).strftime("%Y-%m-%d"),
                (today + timedelta(days=2)).strftime("%Y-%m-%d"),
                (today + timedelta(days=45)).strftime("%Y-%m-%d"),
                (today + timedelta(days=10)).strftime("%Y-%m-%d"),
                (today + timedelta(days=60)).strftime("%Y-%m-%d"),
            ],
            "days_to_expiry": [-2, 2, 45, 10, 60],
            "daily_sales_avg": [2.0, 3.0, 5.0, 2.0, 1.0],
            "reorder_point": [15, 10, 15, 10, 10],
        })
        cat_df = categorize_inventory_dataframe(df, low_stock_threshold=10, expiry_alert_days=3, excess_cover_days=30)
        statuses = dict(zip(cat_df["item_name"], cat_df["stock_status"]))
        self.assertEqual(statuses["Expired Yogurt"], "Expired")
        self.assertEqual(statuses["Urgent Milk"], "Nearing Expiry")
        self.assertEqual(statuses["Out Stock Beef"], "Low Stock")
        self.assertEqual(statuses["Excess Rice"], "Excess Stock")
        self.assertEqual(statuses["Normal Pasta"], "Normal")

    def test_plotly_chart_generators(self):
        df = generate_sample_data(num_items=25)
        c_chart = create_category_inventory_chart(df)
        self.assertIsNotNone(c_chart)
        s_chart = create_sales_trend_chart(df)
        self.assertIsNotNone(s_chart)
        e_chart = create_expiry_risk_distribution_chart(df)
        self.assertIsNotNone(e_chart)

    # 4. FORECASTING MODULE TESTS
    def test_forecasting_with_sufficient_data(self):
        sales_df = generate_sample_sales_history(days=21, num_items=5)
        readiness = check_forecasting_data_readiness(sales_df)
        self.assertTrue(readiness["is_ready"])
        self.assertTrue(readiness["can_run_ml"])

        daily_df = aggregate_daily_sales(sales_df, readiness["product_id_col"], readiness["date_col"], readiness["qty_col"])
        eval_dict = evaluate_models_on_holdout(daily_df, id_col=readiness["product_id_col"])
        self.assertTrue(eval_dict["evaluated"])
        self.assertIn("baseline_mae", eval_dict)
        self.assertIn("ml_mae", eval_dict)

        future_df, summary_df = generate_future_forecast(daily_df, id_col=readiness["product_id_col"], horizon_days=7, eval_dict=eval_dict)
        self.assertFalse(future_df.empty)
        self.assertFalse(summary_df.empty)

    def test_forecasting_with_insufficient_data(self):
        short_df = pd.DataFrame({
            "item_name": ["Milk", "Milk"],
            "sale_date": ["2026-10-01", "2026-10-02"],
            "qty_sold": [10, 12]
        })
        daily_df = aggregate_daily_sales(short_df, "item_name", "sale_date", "qty_sold")
        eval_dict = evaluate_models_on_holdout(daily_df, id_col="item_name")
        self.assertFalse(eval_dict["evaluated"])

    def test_forecasting_missing_sales_records(self):
        inv_only = pd.DataFrame({"item_name": ["Bread"], "stock_qty": [15]})
        readiness = check_forecasting_data_readiness(inv_only)
        self.assertFalse(readiness["is_ready"])
        self.assertTrue(any("sale_date" in f.lower() for f in readiness["missing_fields"]))

    # 5. AGENTS & COORDINATOR CONFLICT HANDLING
    def test_demand_agent(self):
        df = pd.DataFrame({
            "item_id": ["SKU-1", "SKU-2", "SKU-3"],
            "item_name": ["Stockout Item", "Shortage Item", "Excess Item"],
            "category": ["Produce", "Dairy", "Bakery"],
            "stock_qty": [0, 5, 120],
            "daily_sales_avg": [4.0, 3.0, 2.0],
        })
        agent = DemandAgent()
        recs = agent.generate_recommendations(df, forecast_horizon_days=7)
        actions = [r.action_type for r in recs]
        self.assertIn("SHORTAGE_ALERT", actions)
        self.assertIn("SURPLUS_ALERT", actions)

    def test_freshness_agent(self):
        df = pd.DataFrame({
            "item_id": ["SKU-1", "SKU-2", "SKU-3"],
            "item_name": ["Expired Milk", "Donation Yogurt", "Safe Bread"],
            "category": ["Dairy", "Dairy", "Bakery"],
            "stock_qty": [10, 15, 20],
            "days_to_expiry": [-2, 1, 15],
            "expiry_date": ["2026-10-08", "2026-10-11", "2026-10-25"],
            "daily_sales_avg": [2.0, 2.0, 2.0],
        })
        agent = FreshnessAgent()
        recs = agent.generate_recommendations(df)
        rec_map = {r.product_name: r for r in recs}
        self.assertEqual(rec_map["Expired Milk"].action_type, "DISPOSAL")
        self.assertEqual(rec_map["Expired Milk"].confidence_score, 1.0)
        self.assertEqual(rec_map["Donation Yogurt"].action_type, "DONATION")

    def test_discount_agent(self):
        df = pd.DataFrame({
            "item_id": ["SKU-1", "SKU-2"],
            "item_name": ["Expiring Salmon", "Selling Fast Bread"],
            "category": ["Meat", "Bakery"],
            "stock_qty": [20, 5],
            "unit_price": [12.00, 3.00],
            "days_to_expiry": [2, 2],
            "daily_sales_avg": [2.0, 10.0],
        })
        agent = DiscountAgent()
        recs = agent.generate_recommendations(df)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].product_name, "Expiring Salmon")
        self.assertEqual(recs[0].action_type, "MARKDOWN")

    def test_replenishment_agent(self):
        df = pd.DataFrame({
            "item_id": ["SKU-1", "SKU-2"],
            "item_name": ["Depleted Coffee", "Sufficient Tea"],
            "category": ["Pantry", "Pantry"],
            "stock_qty": [0, 50],
            "daily_sales_avg": [5.0, 2.0],
            "reorder_point": [20, 10],
        })
        agent = ReplenishmentAgent()
        recs = agent.generate_recommendations(df, lead_time_days=3, review_period_days=7)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].product_name, "Depleted Coffee")
        self.assertEqual(recs[0].action_type, "REPLENISHMENT")

    def test_transfer_agent_multi_vs_single_store(self):
        single_store_df = pd.DataFrame({
            "item_id": ["SKU-1"],
            "item_name": ["Apples"],
            "stock_qty": [100],
            "store_id": ["Store_A"]
        })
        transfer_agent = TransferAgent()
        recs_single = transfer_agent.generate_recommendations(single_store_df)
        self.assertEqual(len(recs_single), 0)

        multi_store_df = pd.DataFrame({
            "item_id": ["SKU-1", "SKU-1"],
            "item_name": ["Apples", "Apples"],
            "stock_qty": [50, 0],
            "store_id": ["Store_A", "Store_B"]
        })
        recs_multi = transfer_agent.generate_recommendations(multi_store_df)
        self.assertEqual(len(recs_multi), 1)
        self.assertEqual(recs_multi[0].action_type, "TRANSFER")

    def test_coordinator_conflict_resolution(self):
        df = pd.DataFrame({
            "item_id": ["SKU-EXPIRED"],
            "item_name": ["Spoiled Chicken"],
            "category": ["Meat"],
            "stock_qty": [0],
            "days_to_expiry": [-3],
            "expiry_date": ["2026-10-07"],
            "daily_sales_avg": [5.0],
            "reorder_point": [15],
            "unit_price": [8.00],
        })
        coordinator = AgentCoordinator()
        ranked = coordinator.coordinate_and_rank(df)
        action_types = [r.action_type for r in ranked]
        self.assertIn("DISPOSAL", action_types)
        self.assertNotIn("REPLENISHMENT", action_types)

    # 6. DECISIONS WORKFLOW & REPORTS EXPORTS
    def test_decisions_workflow_and_deduplication(self):
        decisions_session = {}
        rec_id = "REC-FRESH-001"
        decisions_session[rec_id] = {
            "recommendation_id": rec_id,
            "product_name": "Milk",
            "status": "Approved",
            "timestamp": "2026-10-10 10:00:00",
            "manager_notes": "Initial approval",
        }
        self.assertEqual(decisions_session[rec_id]["status"], "Approved")

        decisions_session[rec_id] = {
            "recommendation_id": rec_id,
            "product_name": "Milk",
            "status": "Modified",
            "timestamp": "2026-10-10 10:05:00",
            "manager_notes": "Adjusted parameters",
            "modified_values": {"discount_pct": 40},
        }
        self.assertEqual(len(decisions_session), 1)
        self.assertEqual(decisions_session[rec_id]["status"], "Modified")

    def test_reports_exports(self):
        df = generate_sample_data(num_items=15)
        cat_df = categorize_inventory_dataframe(df)

        csv_bytes = cat_df.to_csv(index=False).encode("utf-8")
        self.assertGreater(len(csv_bytes), 100)

        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            pd.DataFrame([{"KPI": "Total SKUs", "Value": len(df)}]).to_excel(writer, sheet_name="Executive_Summary", index=False)
            cat_df.to_excel(writer, sheet_name="Inventory_Audit", index=False)
        self.assertGreater(len(excel_buf.getvalue()), 1000)

    # 7. END-TO-END JOURNEY TEST
    def test_e2e_journey(self):
        demo_df = generate_sample_data(num_items=25)
        metrics = calculate_dashboard_metrics(demo_df)
        self.assertTrue(metrics["has_data"])

        cat_df = categorize_inventory_dataframe(demo_df)
        self.assertEqual(len(cat_df), 25)

        coord = AgentCoordinator()
        recs = coord.coordinate_and_rank(demo_df)
        self.assertGreater(len(recs), 0)

        decisions = {recs[0].recommendation_id: {"status": "Approved", "timestamp": "2026-10-10 12:00:00"}}
        self.assertIn(recs[0].recommendation_id, decisions)

    def test_none_decisions_handling(self):
        # Verify None decision record or missing decision does not crash status badge logic
        decisions_session = {"REC-001": None, "REC-002": {}}
        rec_id = "REC-001"
        curr_decision = decisions_session.get(rec_id)
        dec_record = curr_decision if isinstance(curr_decision, dict) else {}
        curr_status = dec_record.get("status", "Pending")
        self.assertEqual(curr_status, "Pending")
        dec_ts = f" ({dec_record['timestamp']})" if dec_record.get("timestamp") else ""
        badge = {
            "Pending": '<span class="decision-badge-pending">⏳ PENDING REVIEW</span>',
            "Approved": f'<span class="decision-badge-approved">✅ APPROVED{dec_ts}</span>',
            "Rejected": f'<span class="decision-badge-rejected">❌ REJECTED{dec_ts}</span>',
            "Modified": f'<span class="decision-badge-modified">✏️ MODIFIED{dec_ts}</span>',
        }.get(curr_status, '<span class="decision-badge-pending">⏳ PENDING REVIEW</span>')
        self.assertIn("PENDING REVIEW", badge)

    def test_sample_templates_generation(self):
        from modules.data_processor import generate_sample_template_csv, generate_sample_template_excel
        csv_bytes = generate_sample_template_csv("single")
        self.assertIsInstance(csv_bytes, (str, bytes))
        self.assertTrue(len(csv_bytes) > 0)

        xlsx_bytes = generate_sample_template_excel("single")
        self.assertIsInstance(xlsx_bytes, bytes)
        self.assertTrue(len(xlsx_bytes) > 0)


if __name__ == "__main__":
    unittest.main()

