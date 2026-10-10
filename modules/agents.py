"""
Zero-Waste Grocery Multi-Agent Decision Engine

Implements modular, deterministic Python decision agents for inventory management:
1. Demand Agent: Shortage and excess stock identification.
2. Freshness Agent: Spoilage risks and shelf-life monitoring.
3. Discount Agent: Dynamic markdown and clearance suggestions.
4. Replenishment Agent: Optimal reorder quantity estimation.
5. Transfer Agent: Inter-store stock balance (multi-store only).
6. Agent Coordinator: Conflict resolution, business rules, and ranking.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


@dataclass
class AgentRecommendation:
    """
    Standard structured recommendation schema returned by all agents.
    Every recommendation includes explicit supporting evidence and data integrity warnings.
    """
    recommendation_id: str
    agent_name: str
    product_id: str
    product_name: str
    category: str
    proposed_action: str
    action_type: str  # DISPOSAL, DONATION, MARKDOWN, REPLENISHMENT, TRANSFER, SHORTAGE_ALERT, SURPLUS_ALERT, HOLD
    urgency: str      # CRITICAL, HIGH, MEDIUM, LOW
    priority_score: float  # 0.0 to 100.0
    reason: str
    supporting_data: Dict[str, Any]
    missing_data_warnings: List[str]
    expected_benefit: str = ""
    quantity_or_discount: Optional[str] = None
    confidence_score: Optional[float] = None  # Float (0.0 to 1.0) ONLY when defensible calculation exists; else None
    confidence_basis: Optional[str] = None    # Mathematical basis or explanation of qualitative risk level
    llm_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseAgent:
    """Base class for all grocery decision agents."""

    def __init__(self, name: str):
        self.name = name

    def generate_recommendations(self, df: pd.DataFrame, **kwargs) -> List[AgentRecommendation]:
        raise NotImplementedError("Subclasses must implement generate_recommendations")


class DemandAgent(BaseAgent):
    """
    1. Demand Agent:
    Uses available demand forecasts and sales velocity to identify likely
    stock shortages (stockouts) or capital-draining excess inventory.
    """

    def __init__(self, name: str = "Demand Agent"):
        super().__init__(name)

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        forecast_df: Optional[pd.DataFrame] = None,
        forecast_horizon_days: int = 7,
        excess_threshold_ratio: float = 2.5,
    ) -> List[AgentRecommendation]:
        recommendations = []
        if df is None or df.empty:
            return recommendations

        today = datetime.today().date()
        rec_counter = 1

        # Build demand lookup if forecast_df provided
        forecast_demand_map = {}
        if forecast_df is not None and not forecast_df.empty:
            for _, r in forecast_df.iterrows():
                key = str(r.get("item_id", r.get("item_name", "")))
                demand_val = r.get("recommended_total_demand", r.get("projected_total_demand", r.get("recommended_forecast")))
                if pd.notnull(demand_val):
                    forecast_demand_map[key] = float(demand_val)

        for _, row in df.iterrows():
            item_id = str(row.get("item_id", f"ITEM-{rec_counter}"))
            item_name = str(row.get("item_name", item_id))
            category = str(row.get("category", "General"))
            warnings = []

            # 1. Resolve Current Stock
            stock = row.get("stock_qty")
            if pd.isnull(stock):
                continue  # Cannot evaluate demand without stock
            stock = float(stock)

            # 2. Resolve Daily Velocity / Forecast Demand
            projected_demand = forecast_demand_map.get(item_id, forecast_demand_map.get(item_name))
            daily_sales = row.get("daily_sales_avg")
            if pd.isnull(daily_sales) and "qty_sold" in row and pd.notnull(row["qty_sold"]):
                daily_sales = float(row["qty_sold"]) / 7.0

            if projected_demand is None:
                if pd.notnull(daily_sales) and float(daily_sales) > 0:
                    projected_demand = round(float(daily_sales) * forecast_horizon_days, 1)
                else:
                    warnings.append("No demand forecast or sales velocity available; demand risk could not be assessed.")
                    continue  # Do not force recommendations without data

            projected_demand = max(0.1, float(projected_demand))
            cover_ratio = round(stock / projected_demand, 2)

            # Rule A: Stockout / Depleted Stock
            if stock == 0:
                recommendations.append(
                    AgentRecommendation(
                        recommendation_id=f"REC-DEMAND-{rec_counter:03d}",
                        agent_name=self.name,
                        product_id=item_id,
                        product_name=item_name,
                        category=category,
                        proposed_action=f"Urgent Stockout Alert: Depleted Stock ({projected_demand:.0f} units needed)",
                        action_type="SHORTAGE_ALERT",
                        urgency="CRITICAL",
                        priority_score=92.0,
                        reason=f"Item is completely out of stock with expected customer demand of {projected_demand:.0f} units over the next {forecast_horizon_days} days.",
                        supporting_data={
                            "current_stock": stock,
                            "forecast_horizon_days": forecast_horizon_days,
                            "projected_demand": projected_demand,
                            "deficit_units": projected_demand,
                        },
                        missing_data_warnings=warnings,
                        expected_benefit=f"Prevents customer stockouts and recaptures estimated sales of {projected_demand:.0f} units over {forecast_horizon_days} days.",
                        quantity_or_discount=f"{projected_demand:.0f} Units Needed",
                        confidence_score=None,
                        confidence_basis="Qualitative Risk: CRITICAL (Complete on-shelf stock depletion)",
                    )
                )
                rec_counter += 1

            # Rule B: Imminent Shortage (Stock < Forecast Demand)
            elif stock < projected_demand:
                deficit = round(projected_demand - stock, 1)
                days_until_stockout = round((stock / (projected_demand / forecast_horizon_days)), 1) if projected_demand > 0 else 0.0

                recommendations.append(
                    AgentRecommendation(
                        recommendation_id=f"REC-DEMAND-{rec_counter:03d}",
                        agent_name=self.name,
                        product_id=item_id,
                        product_name=item_name,
                        category=category,
                        proposed_action=f"Imminent Shortage Warning: Stockout expected in ~{days_until_stockout} days",
                        action_type="SHORTAGE_ALERT",
                        urgency="HIGH",
                        priority_score=78.0,
                        reason=f"Current stock of {int(stock)} units will run out in ~{days_until_stockout} days against projected demand of {projected_demand:.0f} units.",
                        supporting_data={
                            "current_stock": stock,
                            "forecast_horizon_days": forecast_horizon_days,
                            "projected_demand": projected_demand,
                            "projected_deficit": deficit,
                            "days_until_stockout": days_until_stockout,
                        },
                        missing_data_warnings=warnings,
                        expected_benefit=f"Avoids stockout disruption in ~{days_until_stockout} days by proactively covering {deficit:.0f} unit deficit.",
                        quantity_or_discount=f"{deficit:.0f} Units Deficit",
                        confidence_score=None,
                        confidence_basis=f"Qualitative Risk: HIGH (Depletion expected within {days_until_stockout} days)",
                    )
                )
                rec_counter += 1

            # Rule C: Excess Inventory (Stock significantly exceeds forecast)
            elif cover_ratio >= excess_threshold_ratio and stock >= 20:
                surplus_units = round(stock - (projected_demand * 1.2), 0)
                recommendations.append(
                    AgentRecommendation(
                        recommendation_id=f"REC-DEMAND-{rec_counter:03d}",
                        agent_name=self.name,
                        product_id=item_id,
                        product_name=item_name,
                        category=category,
                        proposed_action=f"Excess Inventory Notice: Surplus of ~{int(surplus_units)} units above demand",
                        action_type="SURPLUS_ALERT",
                        urgency="MEDIUM",
                        priority_score=50.0,
                        reason=f"On-hand inventory ({int(stock)} units) covers {cover_ratio}x the projected {forecast_horizon_days}-day demand of {projected_demand:.0f} units.",
                        supporting_data={
                            "current_stock": stock,
                            "forecast_horizon_days": forecast_horizon_days,
                            "projected_demand": projected_demand,
                            "surplus_units": surplus_units,
                            "stock_to_demand_ratio": cover_ratio,
                        },
                        missing_data_warnings=warnings,
                        expected_benefit=f"Frees up working capital and shelf space by reallocating or promoting ~{int(surplus_units)} surplus units.",
                        quantity_or_discount=f"~{int(surplus_units)} Units Surplus",
                        confidence_score=None,
                        confidence_basis=f"Qualitative Risk: MEDIUM (Stock cover {cover_ratio}x exceeds baseline demand)",
                    )
                )
                rec_counter += 1

        return recommendations


class FreshnessAgent(BaseAgent):
    """
    2. Freshness Agent:
    Uses expiry dates, stock quantities, and available sales velocity to
    identify spoilage risks and prevent perishable food waste.
    """

    def __init__(self, name: str = "Freshness Agent"):
        super().__init__(name)

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        critical_expiry_days: int = 2,
        warning_expiry_days: int = 5,
    ) -> List[AgentRecommendation]:
        recommendations = []
        if df is None or df.empty:
            return recommendations

        today = datetime.today().date()
        rec_counter = 1

        for _, row in df.iterrows():
            item_id = str(row.get("item_id", f"ITEM-{rec_counter}"))
            item_name = str(row.get("item_name", item_id))
            category = str(row.get("category", "General"))
            warnings = []

            # 1. Resolve Days to Expiry (Honest check: do not assume non-perishables expire)
            days = row.get("days_to_expiry")
            exp_date_val = row.get("expiry_date")

            if pd.isnull(days) and pd.notnull(exp_date_val):
                try:
                    parsed_dt = pd.to_datetime(exp_date_val).date()
                    days = (parsed_dt - today).days
                except Exception:
                    days = None

            if pd.isnull(days):
                continue  # Item is non-perishable or has no expiry date recorded; do not force recommendations

            days = int(days)
            stock = row.get("stock_qty")
            stock = float(stock) if pd.notnull(stock) else 0.0

            # 2. Resolve daily velocity
            daily_sales = row.get("daily_sales_avg")
            if pd.isnull(daily_sales) and "qty_sold" in row and pd.notnull(row["qty_sold"]):
                daily_sales = float(row["qty_sold"]) / 7.0
            daily_sales = float(daily_sales) if pd.notnull(daily_sales) else None

            if daily_sales is None:
                warnings.append("Sales velocity not recorded; sell-through risk estimated from stock and days only.")

            # Rule 1: ALREADY EXPIRED (Highest urgency in system)
            if days < 0:
                recommendations.append(
                    AgentRecommendation(
                        recommendation_id=f"REC-FRESH-{rec_counter:03d}",
                        agent_name=self.name,
                        product_id=item_id,
                        product_name=item_name,
                        category=category,
                        proposed_action="Immediate Shelf Disposal: Remove Expired Product from Sales Floor",
                        action_type="DISPOSAL",
                        urgency="CRITICAL",
                        priority_score=100.0,
                        reason=f"Product expired {abs(days)} day(s) ago (Expiry Date: {exp_date_val}). Must be removed immediately for food safety compliance.",
                        supporting_data={
                            "days_to_expiry": days,
                            "expiry_date": str(exp_date_val),
                            "stock_qty": stock,
                            "units_at_risk": stock,
                        },
                        missing_data_warnings=warnings,
                        expected_benefit="Guarantees food safety compliance, prevents public health liability, and clears expired stock.",
                        quantity_or_discount=f"{int(stock)} Units Disposal",
                        confidence_score=1.0,
                        confidence_basis="Defensible Certainty: 100% (Verified calendar expiration date on file)",
                    )
                )
                rec_counter += 1

            # Rule 2: CRITICAL EXPIRY (0 to 2 days) -> Food Bank Donation or Immediate Flash Sale
            elif 0 <= days <= critical_expiry_days:
                expected_sales = round(daily_sales * max(1, days), 1) if daily_sales else 0.0
                unsellable_surplus = max(0.0, stock - expected_sales) if daily_sales else stock

                action_title = "Route to Food Bank Charity Donation" if days <= 1 else "Apply Immediate Flash Markdown or Food Bank Pickup"
                action_type = "DONATION" if days <= 1 else "MARKDOWN"

                benefit_text = (
                    f"Diverts {int(stock)} perishable units to charity food rescue, providing meals and qualifying for tax deductions."
                    if days <= 1
                    else f"Accelerates sell-through of {int(stock)} perishable units before total shelf expiration."
                )

                recommendations.append(
                    AgentRecommendation(
                        recommendation_id=f"REC-FRESH-{rec_counter:03d}",
                        agent_name=self.name,
                        product_id=item_id,
                        product_name=item_name,
                        category=category,
                        proposed_action=f"{action_title} ({int(stock)} units remaining)",
                        action_type=action_type,
                        urgency="CRITICAL",
                        priority_score=94.0,
                        reason=f"Expires in {days} day(s) on {exp_date_val}. At current sales rate, {unsellable_surplus:.0f} units will expire unsold unless action is taken.",
                        supporting_data={
                            "days_to_expiry": days,
                            "expiry_date": str(exp_date_val),
                            "stock_qty": stock,
                            "daily_sales": daily_sales,
                            "unsellable_units_at_risk": unsellable_surplus,
                        },
                        missing_data_warnings=warnings,
                        expected_benefit=benefit_text,
                        quantity_or_discount=f"{int(stock)} Units Donation" if days <= 1 else f"{int(stock)} Units Clearance",
                        confidence_score=None,
                        confidence_basis=f"Qualitative Risk: CRITICAL (Imminent expiry within {days} day(s) <= 2-day threshold)",
                    )
                )
                rec_counter += 1

            # Rule 3: WARNING EXPIRY (3 to 5 days) -> Spoilage Vulnerability
            elif critical_expiry_days < days <= warning_expiry_days:
                cover_days = round(stock / daily_sales, 1) if (daily_sales and daily_sales > 0) else 999.0

                if cover_days > days:
                    surplus = round(stock - (daily_sales * days), 1) if daily_sales else stock
                    recommendations.append(
                        AgentRecommendation(
                            recommendation_id=f"REC-FRESH-{rec_counter:03d}",
                            agent_name=self.name,
                            product_id=item_id,
                            product_name=item_name,
                            category=category,
                            proposed_action=f"Shelf-Life Alert: High Spoilage Risk for ~{int(surplus)} Units",
                            action_type="SURPLUS_ALERT",
                            urgency="HIGH",
                            priority_score=75.0,
                            reason=f"Stock cover ({cover_days} days) exceeds remaining shelf life ({days} days). Expecting {surplus:.0f} units of landfill waste without promo.",
                            supporting_data={
                                "days_to_expiry": days,
                                "expiry_date": str(exp_date_val),
                                "stock_qty": stock,
                                "stock_cover_days": cover_days,
                                "daily_sales": daily_sales,
                                "projected_waste_units": surplus,
                            },
                            missing_data_warnings=warnings,
                            expected_benefit=f"Proactively prevents ~{int(surplus)} units from deteriorating into landfill waste.",
                            quantity_or_discount=f"~{int(surplus)} Units at Risk",
                            confidence_score=None,
                            confidence_basis=f"Qualitative Risk: HIGH (Stock cover {cover_days} days > remaining shelf life {days} days)",
                        )
                    )
                    rec_counter += 1

        return recommendations


class DiscountAgent(BaseAgent):
    """
    3. Discount Agent:
    Suggests deterministic price discounts for products at risk of expiring
    unsold, while preserving margins when items will sell out naturally.
    """

    def __init__(self, name: str = "Discount Agent"):
        super().__init__(name)

    def generate_recommendations(self, df: pd.DataFrame) -> List[AgentRecommendation]:
        recommendations = []
        if df is None or df.empty:
            return recommendations

        rec_counter = 1

        for _, row in df.iterrows():
            item_id = str(row.get("item_id", f"ITEM-{rec_counter}"))
            item_name = str(row.get("item_name", item_id))
            category = str(row.get("category", "General"))
            warnings = []

            # Check days to expiry
            days = row.get("days_to_expiry")
            if pd.isnull(days) and "expiry_date" in row and pd.notnull(row["expiry_date"]):
                try:
                    days = (pd.to_datetime(row["expiry_date"]).date() - datetime.today().date()).days
                except Exception:
                    days = None

            if pd.isnull(days):
                continue  # Non-perishable / no expiry date

            days = int(days)
            # Expired food must NOT be discounted!
            if days < 0:
                continue

            stock = float(row.get("stock_qty", 0))
            if stock <= 0:
                continue

            # Price check
            unit_price = row.get("unit_price")
            if pd.isnull(unit_price):
                warnings.append("Unit price not provided; discount percentage suggested without monetary revenue calculation.")
                unit_price = 0.0
            else:
                unit_price = float(unit_price)

            # Sales velocity check
            daily_sales = row.get("daily_sales_avg")
            if pd.isnull(daily_sales) and "qty_sold" in row and pd.notnull(row["qty_sold"]):
                daily_sales = float(row["qty_sold"]) / 7.0
            daily_sales = float(daily_sales) if pd.notnull(daily_sales) else None

            # Check natural sell-through: if stock will sell out naturally, do NOT discount!
            if daily_sales and (stock <= daily_sales * days):
                continue  # No discount needed; natural sell-through expected

            # Deterministic discount ladder
            discount_pct = 0
            markdown_tier = ""
            urgency = "LOW"
            priority = 40.0

            if 1 <= days <= 2:
                discount_pct = 50
                markdown_tier = "⚡ 50% Quick-Sale Clearance Markdown"
                urgency = "HIGH"
                priority = 86.0
            elif 3 <= days <= 4:
                discount_pct = 30
                markdown_tier = "🏷️ 30% Dynamic Markdown"
                urgency = "MEDIUM"
                priority = 68.0
            elif 5 <= days <= 7:
                discount_pct = 15
                markdown_tier = "🌱 15% Early Eco-Promo Discount"
                urgency = "LOW"
                priority = 48.0

            if discount_pct > 0:
                new_price = round(unit_price * (1 - discount_pct / 100.0), 2)
                potential_recovered_revenue = round(new_price * stock, 2)

                recommendations.append(
                    AgentRecommendation(
                        recommendation_id=f"REC-DISCOUNT-{rec_counter:03d}",
                        agent_name=self.name,
                        product_id=item_id,
                        product_name=item_name,
                        category=category,
                        proposed_action=f"Apply {discount_pct}% Markdown (New Price: ${new_price:.2f})" if unit_price > 0 else f"Apply {discount_pct}% Markdown",
                        action_type="MARKDOWN",
                        urgency=urgency,
                        priority_score=priority,
                        reason=f"Product expires in {days} days with stock ({int(stock)} units) exceeding expected natural sell-through. Applying a {discount_pct}% markdown stimulates immediate sales.",
                        supporting_data={
                            "days_to_expiry": days,
                            "original_price": unit_price,
                            "discount_pct": discount_pct,
                            "discounted_price": new_price,
                            "stock_qty": stock,
                            "estimated_revenue_recovered": potential_recovered_revenue,
                            "markdown_tier": markdown_tier,
                        },
                        missing_data_warnings=warnings,
                        expected_benefit=f"Recovers estimated ${potential_recovered_revenue:,.2f} in revenue and clears {int(stock)} units before expiration." if unit_price > 0 else f"Accelerates sell-through of {int(stock)} units before shelf expiry.",
                        quantity_or_discount=f"{discount_pct}% Markdown (${new_price:.2f})" if unit_price > 0 else f"{discount_pct}% Markdown",
                        confidence_score=None,
                        confidence_basis=f"Qualitative Risk: {urgency} (Rule-based discount ladder: {markdown_tier})",
                    )
                )
                rec_counter += 1

        return recommendations


class ReplenishmentAgent(BaseAgent):
    """
    4. Replenishment Agent:
    Estimates suitable reorder quantities using on-hand inventory,
    lead time demand, and safety stock rules.
    """

    def __init__(self, name: str = "Replenishment Agent"):
        super().__init__(name)

    def generate_recommendations(
        self,
        df: pd.DataFrame,
        lead_time_days: int = 3,
        review_period_days: int = 7,
        min_order_qty: int = 10,
    ) -> List[AgentRecommendation]:
        recommendations = []
        if df is None or df.empty:
            return recommendations

        rec_counter = 1

        for _, row in df.iterrows():
            item_id = str(row.get("item_id", f"ITEM-{rec_counter}"))
            item_name = str(row.get("item_name", item_id))
            category = str(row.get("category", "General"))
            warnings = []

            stock = row.get("stock_qty")
            if pd.isnull(stock):
                continue
            stock = float(stock)

            # Sales velocity
            daily_sales = row.get("daily_sales_avg")
            if pd.isnull(daily_sales) and "qty_sold" in row and pd.notnull(row["qty_sold"]):
                daily_sales = float(row["qty_sold"]) / 7.0

            if pd.isnull(daily_sales) or float(daily_sales) <= 0:
                warnings.append("Daily sales rate not tracked; reorder calculation based on static reorder threshold only.")
                daily_sales = 5.0  # Conservative baseline fallback

            daily_sales = float(daily_sales)

            # Check custom reorder point from dataset
            custom_rp = row.get("reorder_point")
            if pd.notnull(custom_rp):
                reorder_threshold = float(custom_rp)
                safety_stock = max(5.0, round(reorder_threshold - (daily_sales * lead_time_days), 1))
            else:
                safety_stock = round(daily_sales * 2.0, 1)
                reorder_threshold = round((daily_sales * lead_time_days) + safety_stock, 1)

            # Trigger condition: Stock is at or below threshold
            if stock <= reorder_threshold:
                target_stock = (daily_sales * (lead_time_days + review_period_days)) + safety_stock
                suggested_order = int(max(min_order_qty, round(target_stock - stock)))

                # Check if item has expired or nearing expiry; if so, flag constraint
                days = row.get("days_to_expiry")
                if pd.notnull(days) and int(days) <= lead_time_days + 1:
                    warnings.append(f"Caution: Existing batch expires in {int(days)} days. Ensure old stock is cleared before new delivery arrives.")

                is_stockout = (stock == 0)
                urgency = "CRITICAL" if is_stockout else ("HIGH" if stock <= 5 else "MEDIUM")
                priority = 91.0 if is_stockout else 68.0

                recommendations.append(
                    AgentRecommendation(
                        recommendation_id=f"REC-REPLENISH-{rec_counter:03d}",
                        agent_name=self.name,
                        product_id=item_id,
                        product_name=item_name,
                        category=category,
                        proposed_action=f"Place Replenishment Purchase Order for {suggested_order} Units",
                        action_type="REPLENISHMENT",
                        urgency=urgency,
                        priority_score=priority,
                        reason=f"Current stock ({int(stock)}) is at or below reorder threshold ({reorder_threshold:.0f}). An order of {suggested_order} units covers {lead_time_days}-day lead time plus {review_period_days}-day sales.",
                        supporting_data={
                            "current_stock": stock,
                            "daily_sales_avg": daily_sales,
                            "reorder_threshold": reorder_threshold,
                            "lead_time_days": lead_time_days,
                            "safety_stock": safety_stock,
                            "suggested_order_qty": suggested_order,
                        },
                        missing_data_warnings=warnings,
                        expected_benefit=f"Secures supply for {lead_time_days}-day supplier lead time + {review_period_days}-day sales cycle while maintaining {safety_stock:.0f} units safety stock.",
                        quantity_or_discount=f"Order {suggested_order} Units",
                        confidence_score=1.0 if is_stockout else None,
                        confidence_basis="Defensible Certainty: 100% (Confirmed zero physical stock on hand)" if is_stockout else f"Qualitative Risk: {urgency} (Stock {int(stock)} <= reorder threshold {reorder_threshold:.0f})",
                    )
                )
                rec_counter += 1

        return recommendations


class TransferAgent(BaseAgent):
    """
    5. Transfer Agent:
    Suggests possible inter-store stock transfers ONLY when data for multiple
    stores is present in the dataset. Never invents transfers for single-store datasets.
    """

    def __init__(self, name: str = "Transfer Agent"):
        super().__init__(name)

    def generate_recommendations(self, df: pd.DataFrame) -> List[AgentRecommendation]:
        recommendations = []
        if df is None or df.empty:
            return recommendations

        # Check multi-store condition honestly
        has_store_col = any(col in df.columns for col in ["store_id", "store_location", "branch_id", "location"])
        store_col = next((c for c in ["store_id", "store_location", "branch_id", "location"] if c in df.columns), None)

        if not has_store_col or store_col is None:
            # Single-store dataset: Do not invent transfers!
            return recommendations

        unique_stores = df[store_col].dropna().unique()
        if len(unique_stores) < 2:
            # Only 1 store in dataset: Cannot perform transfer
            return recommendations

        rec_counter = 1
        # Multi-store balancing logic
        for item_id, group in df.groupby("item_id"):
            item_name = str(group["item_name"].iloc[0]) if "item_name" in group.columns else str(item_id)
            category = str(group["category"].iloc[0]) if "category" in group.columns else "General"

            # Check stores with surplus vs stores with stockout
            surplus_stores = group[group["stock_qty"] > 30]
            deficit_stores = group[group["stock_qty"] <= 5]

            if not surplus_stores.empty and not deficit_stores.empty:
                src_row = surplus_stores.sort_values("stock_qty", ascending=False).iloc[0]
                dst_row = deficit_stores.sort_values("stock_qty", ascending=True).iloc[0]

                src_store = str(src_row[store_col])
                dst_store = str(dst_row[store_col])
                src_stock = float(src_row["stock_qty"])
                dst_stock = float(dst_row["stock_qty"])

                transfer_qty = int(min(round(src_stock * 0.4), 25))
                if transfer_qty >= 5:
                    recommendations.append(
                        AgentRecommendation(
                            recommendation_id=f"REC-TRANSFER-{rec_counter:03d}",
                            agent_name=self.name,
                            product_id=str(item_id),
                            product_name=item_name,
                            category=category,
                            proposed_action=f"Transfer {transfer_qty} Units from Store '{src_store}' to Store '{dst_store}'",
                            action_type="TRANSFER",
                            urgency="HIGH" if dst_stock == 0 else "MEDIUM",
                            priority_score=76.0,
                            reason=f"Store '{src_store}' has surplus ({int(src_stock)} units) while Store '{dst_store}' faces shortage ({int(dst_stock)} units). Inter-store transfer resolves shortage without new purchase order.",
                            supporting_data={
                                "source_store": src_store,
                                "source_stock": src_stock,
                                "destination_store": dst_store,
                                "destination_stock": dst_stock,
                                "recommended_transfer_units": transfer_qty,
                            },
                            missing_data_warnings=[],
                            expected_benefit=f"Fulfills shortage at Store '{dst_store}' without supplier purchasing costs or lead-time delay, utilizing surplus at Store '{src_store}'.",
                            quantity_or_discount=f"Transfer {transfer_qty} Units",
                            confidence_score=None,
                            confidence_basis=f"Qualitative Risk: {'HIGH' if dst_stock == 0 else 'MEDIUM'} (Surplus store stock: {int(src_stock)}, Deficit store stock: {int(dst_stock)})",
                        )
                    )
                    rec_counter += 1

        return recommendations


class AgentCoordinator:
    """
    Coordinates and combines all agent recommendations:
    - Gathers proposals from Demand, Freshness, Discount, Replenishment, Transfer agents.
    - Detects and resolves operational conflicts (e.g. no replenishment on items being cleared).
    - Applies business rules.
    - Produces a ranked list of proposed actions.
    """

    def __init__(self):
        self.demand_agent = DemandAgent()
        self.freshness_agent = FreshnessAgent()
        self.discount_agent = DiscountAgent()
        self.replenishment_agent = ReplenishmentAgent()
        self.transfer_agent = TransferAgent()

    def run_all_agents(
        self,
        df: pd.DataFrame,
        forecast_df: Optional[pd.DataFrame] = None,
        forecast_horizon_days: int = 7,
    ) -> Dict[str, List[AgentRecommendation]]:
        """Executes all 5 agents independently and returns raw outputs per agent."""
        return {
            "Demand Agent": self.demand_agent.generate_recommendations(df, forecast_df=forecast_df, forecast_horizon_days=forecast_horizon_days),
            "Freshness Agent": self.freshness_agent.generate_recommendations(df),
            "Discount Agent": self.discount_agent.generate_recommendations(df),
            "Replenishment Agent": self.replenishment_agent.generate_recommendations(df),
            "Transfer Agent": self.transfer_agent.generate_recommendations(df),
        }

    def coordinate_and_rank(
        self,
        df: pd.DataFrame,
        forecast_df: Optional[pd.DataFrame] = None,
        forecast_horizon_days: int = 7,
    ) -> List[AgentRecommendation]:
        """
        Executes agents, filters conflicts, enforces business constraints,
        and returns a sorted, ranked list of coherent proposed actions.
        """
        agent_outputs = self.run_all_agents(df, forecast_df=forecast_df, forecast_horizon_days=forecast_horizon_days)

        all_recs: List[AgentRecommendation] = []
        for recs in agent_outputs.values():
            all_recs.extend(recs)

        if not all_recs:
            return []

        # -------------------------------------------------------------
        # Conflict Detection & Business Rules Resolution
        # -------------------------------------------------------------
        # Identify items with specific actions
        disposal_items = {r.product_id for r in all_recs if r.action_type == "DISPOSAL"}
        donation_items = {r.product_id for r in all_recs if r.action_type == "DONATION"}
        markdown_items = {r.product_id for r in all_recs if r.action_type == "MARKDOWN"}
        transfer_items = {r.product_id for r in all_recs if r.action_type == "TRANSFER"}

        resolved_recs = []

        for rec in all_recs:
            pid = rec.product_id

            # Conflict Rule 1: Expired Items (DISPOSAL takes absolute precedence)
            if pid in disposal_items and rec.action_type != "DISPOSAL":
                # Suppress markdown, transfer, or replenishment for expired products!
                continue

            # Conflict Rule 2: Items being Donated or Markdown-Cleared (Suppress Replenishment)
            if (pid in donation_items or pid in markdown_items) and rec.action_type == "REPLENISHMENT":
                # Do not place purchase orders for products actively being liquidated due to close shelf-life!
                continue

            # Conflict Rule 3: Redundant Shortage Alert if Replenishment is already proposed
            if rec.action_type == "SHORTAGE_ALERT":
                has_replenish = any(r.product_id == pid and r.action_type == "REPLENISHMENT" for r in all_recs)
                if has_replenish:
                    # Deprioritize shortage alert so replenishment action takes the spotlight
                    rec.priority_score -= 15.0

            resolved_recs.append(rec)

        # -------------------------------------------------------------
        # Ranking Algorithm
        # -------------------------------------------------------------
        # Sort primarily by priority_score descending, then urgency rank
        urgency_weight = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

        resolved_recs.sort(
            key=lambda r: (
                r.priority_score,
                urgency_weight.get(r.urgency, 0),
            ),
            reverse=True,
        )

        return resolved_recs

    def explain_recommendation_deterministically(self, rec: AgentRecommendation) -> str:
        """
        Generates transparent, deterministic natural language explanation of a recommendation
        grounded entirely on supporting_data without hallucinated numbers.
        """
        supp = rec.supporting_data
        name = rec.product_name
        act = rec.action_type

        if act == "DISPOSAL":
            return (
                f"🚨 **Safety Directive:** {name} reached its expiration date on {supp.get('expiry_date')} "
                f"({abs(supp.get('days_to_expiry', 0))} day(s) ago). Selling expired inventory violates food safety guidelines. "
                f"Immediately remove all {supp.get('stock_qty', 0)} shelf units."
            )
        elif act == "DONATION":
            return (
                f"🤝 **Food Rescue Routing:** {name} expires in {supp.get('days_to_expiry')} day(s). "
                f"Natural retail sales cannot clear the remaining {supp.get('stock_qty')} units in time. "
                f"Routing to local charity pickup prevents landfill waste and qualifies for tax deductions."
            )
        elif act == "MARKDOWN":
            return (
                f"🏷️ **Dynamic Pricing:** {name} has {supp.get('stock_qty')} units remaining with {supp.get('days_to_expiry')} days of shelf life. "
                f"A {supp.get('discount_pct')}% discount reduces price from ${supp.get('original_price', 0):.2f} to ${supp.get('discounted_price', 0):.2f}, "
                f"accelerating sales velocity to recover an estimated ${supp.get('estimated_revenue_recovered', 0):,.2f} before expiry."
            )
        elif act == "REPLENISHMENT":
            return (
                f"📦 **Inventory Restock:** Current stock ({supp.get('current_stock', 0)} units) has fallen to or below the "
                f"safety reorder threshold ({supp.get('reorder_threshold', 0):.0f} units). "
                f"Ordering {supp.get('suggested_order_qty')} units covers the {supp.get('lead_time_days', 3)}-day delivery lead time."
            )
        elif act == "TRANSFER":
            return (
                f"🔄 **Network Balance:** Store '{supp.get('destination_store')}' is short on stock, while store "
                f"'{supp.get('source_store')}' has surplus inventory ({supp.get('source_stock', 0)} units). "
                f"Transferring {supp.get('recommended_transfer_units')} units fulfills demand without purchasing new external inventory."
            )
        elif act == "SHORTAGE_ALERT":
            return (
                f"⚠️ **Demand Gap:** Customer demand is projected at {supp.get('projected_demand', 0):.0f} units over "
                f"{supp.get('forecast_horizon_days', 7)} days, but on-hand stock is only {supp.get('current_stock', 0)} units."
            )
        elif act == "SURPLUS_ALERT":
            return (
                f"📈 **Holding Capital Alert:** Current stock of {supp.get('current_stock', 0)} units exceeds expected demand by "
                f"{supp.get('stock_to_demand_ratio', 1.0):.1f}x. Holding surplus ties up working capital."
            )
        else:
            return f"Action proposed: {rec.proposed_action}. Reason: {rec.reason}"
