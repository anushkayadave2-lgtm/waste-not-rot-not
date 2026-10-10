import pandas as pd


def get_discount_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates dynamic price markdown recommendations based on days to expiry.
    """
    if df.empty:
        return pd.DataFrame()

    recommendations = []
    for _, row in df.iterrows():
        days = row.get("days_to_expiry", 99)
        price = row.get("unit_price", 0.0)
        stock = row.get("stock_qty", 0)

        discount_pct = 0
        action = "Regular Price"

        if days <= 1:
            discount_pct = 50
            action = "⚡ Quick-Sale Clearance (50% Off)"
        elif days <= 3:
            discount_pct = 30
            action = "🏷️ Dynamic Markdown (30% Off)"
        elif days <= 5:
            discount_pct = 15
            action = "🌱 Eco-Promo Discount (15% Off)"

        if discount_pct > 0:
            discounted_price = round(price * (1 - discount_pct / 100.0), 2)
            potential_revenue_saved = round(discounted_price * stock, 2)
            recommendations.append(
                {
                    "item_id": row.get("item_id", ""),
                    "item_name": row.get("item_name", ""),
                    "category": row.get("category", ""),
                    "days_to_expiry": days,
                    "stock_qty": stock,
                    "original_price": f"${price:.2f}",
                    "discount_applied": f"{discount_pct}%",
                    "new_price": f"${discounted_price:.2f}",
                    "recommended_action": action,
                    "potential_revenue_saved": potential_revenue_saved,
                }
            )

    return pd.DataFrame(recommendations)


def get_reorder_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recommends smart reorder quantities to prevent both stockouts and overstock waste.
    """
    if df.empty:
        return pd.DataFrame()

    reorders = []
    for _, row in df.iterrows():
        stock = row.get("stock_qty", 0)
        reorder_pt = row.get("reorder_point", 15)
        daily_sales = row.get("daily_sales_avg", 5.0)

        if stock <= reorder_pt:
            suggested_reorder = int(max(10, round(daily_sales * 7 - stock)))
            reorders.append(
                {
                    "item_id": row.get("item_id", ""),
                    "item_name": row.get("item_name", ""),
                    "category": row.get("category", ""),
                    "current_stock": stock,
                    "reorder_threshold": reorder_pt,
                    "daily_sales_avg": daily_sales,
                    "suggested_order_qty": suggested_reorder,
                    "status": "📦 REORDER NEEDED",
                }
            )

    return pd.DataFrame(reorders)


def get_donation_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies inventory eligible for food bank donations (expiring in 1-2 days).
    """
    if df.empty:
        return pd.DataFrame()

    donations = df[df["days_to_expiry"].isin([1, 2])].copy()
    if donations.empty:
        return pd.DataFrame()

    donations["donation_value"] = (donations["stock_qty"] * donations["unit_price"]).round(2)
    donations["tax_deduction_est"] = (donations["donation_value"] * 0.5).round(2)
    donations["action"] = "🤝 Schedule Food Bank Pickup"

    return donations[["item_id", "item_name", "category", "stock_qty", "days_to_expiry", "donation_value", "tax_deduction_est", "action"]]


def get_all_agent_recommendations(df: pd.DataFrame, forecast_df: pd.DataFrame = None, horizon_days: int = 7):
    """Executes all 5 deterministic agents and returns dictionary of recommendations per agent."""
    from modules.agents import AgentCoordinator
    coord = AgentCoordinator()
    return coord.run_all_agents(df, forecast_df=forecast_df, forecast_horizon_days=horizon_days)


def get_ranked_agent_proposals(df: pd.DataFrame, forecast_df: pd.DataFrame = None, horizon_days: int = 7):
    """Executes AgentCoordinator conflict resolution and produces ranked list of structured recommendations."""
    from modules.agents import AgentCoordinator
    coord = AgentCoordinator()
    return coord.coordinate_and_rank(df, forecast_df=forecast_df, forecast_horizon_days=horizon_days)
