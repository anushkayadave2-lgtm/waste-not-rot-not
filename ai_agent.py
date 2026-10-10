from typing import Dict, List
import pandas as pd


class WasteReductionAgent:
    """
    AI Agent responsible for auditing inventory for waste risk, identifying patterns,
    and triggering automated sustainability insights and alerts.
    """

    def __init__(self, name: str = "EcoAgent AI"):
        self.name = name

    def audit_inventory(self, df: pd.DataFrame) -> Dict:
        """Audits inventory to compute risk scores and flag vulnerabilities."""
        if df.empty:
            return {"sustainability_score": 100, "critical_count": 0, "moderate_count": 0}

        critical_items = df[df["days_to_expiry"] <= 3]
        moderate_items = df[(df["days_to_expiry"] > 3) & (df["days_to_expiry"] <= 7) & (df["waste_risk_pct"] > 30.0)]

        high_risk_value = (critical_items["stock_qty"] * critical_items["unit_price"]).sum()
        total_value = (df["stock_qty"] * df["unit_price"]).sum()

        risk_ratio = (high_risk_value / total_value) if total_value > 0 else 0
        sustainability_score = max(0, min(100, round((1.0 - risk_ratio) * 100, 1)))

        return {
            "sustainability_score": sustainability_score,
            "critical_count": len(critical_items),
            "moderate_count": len(moderate_items),
            "at_risk_value_usd": round(float(high_risk_value), 2),
        }

    def generate_alerts(self, df: pd.DataFrame) -> List[Dict]:
        """Generates list of prioritized alerts for store management."""
        alerts = []
        if df.empty:
            return alerts

        for _, row in df.iterrows():
            days = row.get("days_to_expiry", 99)
            stock = row.get("stock_qty", 0)
            item_name = row.get("item_name", "Item")
            category = row.get("category", "General")
            price = row.get("unit_price", 0.0)

            if days <= 2:
                alerts.append(
                    {
                        "severity": "CRITICAL",
                        "item": item_name,
                        "category": category,
                        "message": f"🚨 {stock} units expiring in {days} day(s). Immediate markdown or food bank donation required!",
                        "financial_impact": f"${round(stock * price, 2)}",
                    }
                )
            elif days <= 5 and row.get("waste_risk_pct", 0) > 40:
                alerts.append(
                    {
                        "severity": "WARNING",
                        "item": item_name,
                        "category": category,
                        "message": f"⚠️ High waste risk ({row.get('waste_risk_pct')}%). Stock cover exceeds remaining shelf life.",
                        "financial_impact": f"${round(stock * price, 2)}",
                    }
                )
            elif stock > row.get("reorder_point", 20) * 3:
                alerts.append(
                    {
                        "severity": "INFO",
                        "item": item_name,
                        "category": category,
                        "message": f"ℹ️ Overstock detected ({stock} units). Consider pausing purchase orders.",
                        "financial_impact": f"${round(stock * price, 2)}",
                    }
                )

        return sorted(alerts, key=lambda x: 0 if x["severity"] == "CRITICAL" else (1 if x["severity"] == "WARNING" else 2))

    def get_agent_summary(self, df: pd.DataFrame) -> str:
        """Returns natural language summary of the current inventory state."""
        audit = self.audit_inventory(df)
        score = audit["sustainability_score"]

        if score >= 85:
            return f"🌿 **{self.name} Status:** Inventory health is excellent with a Sustainability Score of {score}%. Expiry risk is minimal."
        elif score >= 65:
            return f"⚡ **{self.name} Status:** Sustainability Score is {score}%. Moderate action required for {audit['critical_count']} items expiring within 3 days."
        else:
            return f"🔥 **{self.name} Status:** URGENT! Sustainability Score dropped to {score}%. ${audit['at_risk_value_usd']} worth of stock is in danger of being wasted."


class LLMRecommendationExplainer:
    """
    Optional natural language explanation engine for agent recommendations.
    Strictly uses deterministic supporting data as ground truth context.
    Never invents facts, changes numerical values, or overrides deterministic rules.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.has_api_key = bool(api_key and api_key.strip())

    def explain(self, recommendation) -> str:
        """
        Explains a structured recommendation using deterministic evidence.
        Works 100% offline without an API key using transparent rule-based templates.
        """
        supp = getattr(recommendation, "supporting_data", {})
        act_type = getattr(recommendation, "action_type", "")
        name = getattr(recommendation, "product_name", "Product")
        reason = getattr(recommendation, "reason", "")
        warnings = getattr(recommendation, "missing_data_warnings", [])

        # Offline deterministic explanation template
        explanation = (
            f"**Action Recommended:** {getattr(recommendation, 'proposed_action', '')}\n\n"
            f"**Evidence & Rationale:** {reason}\n\n"
            f"**Key Supporting Metrics:** "
            + ", ".join([f"{k.replace('_', ' ').title()}: `{v}`" for k, v in supp.items() if not isinstance(v, (dict, list))])
        )

        if warnings:
            explanation += f"\n\n⚠️ **Data Integrity Notice:** {'; '.join(warnings)}"

        return explanation
