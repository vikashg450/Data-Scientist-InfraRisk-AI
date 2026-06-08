import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class FinancialFeaturesCalculator:
    """
    Computes project-level financial features, debt coverage ratios (DSCR, LLCR, PLCR),
    and models the SPV cash waterfall for infrastructure project finance.
    """

    def __init__(self, base_interest_rate: float = 0.05, tax_rate: float = 0.20):
        self.base_interest_rate = base_interest_rate
        self.tax_rate = tax_rate

    @staticmethod
    def calculate_dscr(cfads: float, debt_service: float) -> float:
        """
        Calculates the Debt Service Coverage Ratio (DSCR).
        Formula: DSCR = CFADS / Debt Service
        """
        if debt_service <= 0:
            return 99.0  # High coverage proxy if no debt service
        return max(0.0, cfads / debt_service)

    @staticmethod
    def calculate_npv(cash_flows: List[float], discount_rate: float) -> float:
        """
        Computes the Net Present Value (NPV) of a series of cash flows.
        """
        return sum(cf / ((1 + discount_rate) ** t) for t, cf in enumerate(cash_flows, start=1))

    def calculate_llcr(self, cfads_remaining: List[float], outstanding_debt: float, discount_rate: float) -> float:
        """
        Calculates the Loan Life Coverage Ratio (LLCR).
        Formula: LLCR = NPV(CFADS over remaining loan life) / Outstanding Debt Balance
        """
        if outstanding_debt <= 0:
            return 99.0
        npv_cfads = self.calculate_npv(cfads_remaining, discount_rate)
        return max(0.0, npv_cfads / outstanding_debt)

    def calculate_plcr(self, cfads_remaining_project: List[float], outstanding_debt: float, discount_rate: float) -> float:
        """
        Calculates the Project Life Coverage Ratio (PLCR).
        Formula: PLCR = NPV(CFADS over remaining project life) / Outstanding Debt Balance
        """
        if outstanding_debt <= 0:
            return 99.0
        npv_cfads_project = self.calculate_npv(cfads_remaining_project, discount_rate)
        return max(0.0, npv_cfads_project / outstanding_debt)

    def simulate_spv_waterfall(
        self,
        investment_value: float,
        debt_value: float,
        concession_period: int,
        base_dscr: float,
        sector: str,
        interest_rate: float = None,
        sweep_pct: float = 0.50,
        construction_period: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Simulates the SPV cash waterfall over the project's concession period.
        Waterfall Priority levels:
        1. Revenue -> Operating Expenses (OpEx)
        2. Taxes
        3. Senior Debt Service (Interest + Scheduled Principal)
        4. Debt Service Reserve Account (DSRA) top-up
        5. Maintenance Reserve Account (MRA) top-up
        6. Cash Sweep (if active)
        7. Subordinated Debt Service (if any)
        8. Equity Distributions (Dividends)
        """
        interest_rate = interest_rate if interest_rate is not None else self.base_interest_rate
        operational_years = concession_period - construction_period
        
        # Debt tenor is typically concession tail shorter (e.g. 5 years tail)
        debt_tenor = max(1, operational_years - 5)
        
        # Scheduled principal amortization (straight line for simplicity)
        scheduled_principal_per_year = debt_value / debt_tenor if debt_tenor > 0 else 0
        
        outstanding_debt = debt_value
        dsra_balance = 0.0
        mra_balance = 0.0
        
        waterfall_records = []
        
        for year in range(1, concession_period + 1):
            record = {
                "year": year,
                "revenue": 0.0,
                "opex": 0.0,
                "ebitda": 0.0,
                "interest": 0.0,
                "scheduled_principal": 0.0,
                "prepayment": 0.0,
                "debt_service": 0.0,
                "taxes": 0.0,
                "cfads": 0.0,
                "dsra_balance": dsra_balance,
                "mra_balance": mra_balance,
                "equity_distribution": 0.0,
                "outstanding_debt": outstanding_debt,
                "dscr": 0.0,
                "llcr": 0.0,
                "plcr": 0.0,
                "phase": "construction" if year <= construction_period else "operations"
            }
            
            if record["phase"] == "construction":
                # Capitalized Interest during construction (IDC)
                idc = outstanding_debt * interest_rate
                outstanding_debt += idc
                record["outstanding_debt"] = outstanding_debt
                record["interest"] = idc
                waterfall_records.append(record)
                continue
                
            # Operations Phase
            op_year = year - construction_period
            
            # 1. Estimate Interest and Scheduled Principal
            interest = outstanding_debt * interest_rate
            scheduled_principal = scheduled_principal_per_year if op_year <= debt_tenor else 0.0
            
            # If remaining debt is less than scheduled principal
            if scheduled_principal > outstanding_debt:
                scheduled_principal = outstanding_debt
                
            debt_service = interest + scheduled_principal
            
            # 2. Estimate CFADS needed based on base DSCR
            target_cfads = debt_service * base_dscr if debt_service > 0 else investment_value * 0.10
            
            # Revenue & OpEx model based on sector
            if sector == "Energy":
                opex_pct = 0.45
            elif sector == "Transport":
                opex_pct = 0.30
            elif sector == "Water and sewerage":
                opex_pct = 0.35
            else:  # Telecom
                opex_pct = 0.25
                
            # CFADS is EBITDA - Taxes.
            # EBITDA = Revenue - OpEx = Revenue * (1 - opex_pct)
            # Let's backward solve Revenue from target CFADS
            # We assume a simplified EBITDA to CFADS conversion for target
            ebitda_target = target_cfads / (1.0 - self.tax_rate)
            revenue = ebitda_target / (1.0 - opex_pct)
            opex = revenue * opex_pct
            ebitda = revenue - opex
            
            # 3. Calculate actual Taxes (accounting for interest tax shield and depreciation)
            depreciation = investment_value / operational_years
            taxable_income = max(0.0, ebitda - interest - depreciation)
            taxes = taxable_income * self.tax_rate
            
            cfads = max(0.0, ebitda - taxes)
            record["revenue"] = round(revenue, 3)
            record["opex"] = round(opex, 3)
            record["ebitda"] = round(ebitda, 3)
            record["taxes"] = round(taxes, 3)
            record["cfads"] = round(cfads, 3)
            record["interest"] = round(interest, 3)
            record["scheduled_principal"] = round(scheduled_principal, 3)
            
            # 4. Debt Service Priority
            actual_debt_service = min(cfads, debt_service)
            cf_after_debt_service = cfads - actual_debt_service
            record["debt_service"] = round(actual_debt_service, 3)
            
            # DSCR calculation
            record["dscr"] = round(self.calculate_dscr(cfads, debt_service), 3)
            
            # Update outstanding debt
            paid_principal = min(outstanding_debt, scheduled_principal)
            outstanding_debt -= paid_principal
            
            # 5. DSRA (Debt Service Reserve Account) Top-up
            # Target is 6 months of senior debt service (0.5 * debt_service)
            dsra_target = 0.5 * debt_service
            dsra_drawdown = 0.0
            
            # If CFADS was not enough to pay debt service, draw from DSRA
            if cfads < debt_service:
                needed = debt_service - cfads
                dsra_drawdown = min(dsra_balance, needed)
                dsra_balance -= dsra_drawdown
                actual_debt_service += dsra_drawdown
                record["debt_service"] = round(actual_debt_service, 3)
                # Re-calculate DSCR if we had a shortfall but covered by DSRA, the technical DSCR is still based on cash flow
                
            # If we have excess cash, replenish DSRA
            dsra_replenishment = 0.0
            if cf_after_debt_service > 0 and dsra_balance < dsra_target:
                needed = dsra_target - dsra_balance
                dsra_replenishment = min(cf_after_debt_service, needed)
                dsra_balance += dsra_replenishment
                cf_after_debt_service -= dsra_replenishment
                
            # 6. MRA (Maintenance Reserve Account) Top-up
            # Sinking fund for lifecycle maintenance (say 2% of investment value target)
            mra_target = investment_value * 0.02
            mra_replenishment = 0.0
            if cf_after_debt_service > 0 and mra_balance < mra_target:
                needed = mra_target - mra_balance
                mra_replenishment = min(cf_after_debt_service, needed)
                mra_balance += mra_replenishment
                cf_after_debt_service -= mra_replenishment
                
            # 7. Cash Sweep
            prepayment = 0.0
            if cf_after_debt_service > 0 and outstanding_debt > 0 and sweep_pct > 0:
                sweep_cash = cf_after_debt_service * sweep_pct
                prepayment = min(outstanding_debt, sweep_cash)
                outstanding_debt -= prepayment
                cf_after_debt_service -= prepayment
                record["prepayment"] = round(prepayment, 3)
                
            # 8. Subordinated Debt and Equity Distributions
            equity_dist = max(0.0, cf_after_debt_service)
            record["equity_distribution"] = round(equity_dist, 3)
            
            record["dsra_balance"] = round(dsra_balance, 3)
            record["mra_balance"] = round(mra_balance, 3)
            record["outstanding_debt"] = round(outstanding_debt, 3)
            
            waterfall_records.append(record)
            
        # Add LLCR and PLCR to operational records
        for i, rec in enumerate(waterfall_records):
            if rec["phase"] == "construction":
                continue
                
            current_yr = rec["year"]
            out_debt = rec["outstanding_debt"]
            
            # Remaining CFADS over loan life
            # Loan life ends at construction_period + debt_tenor
            loan_end_year = construction_period + debt_tenor
            cfads_loan_life = [
                r["cfads"] for r in waterfall_records 
                if current_yr <= r["year"] <= loan_end_year and r["phase"] == "operations"
            ]
            rec["llcr"] = round(self.calculate_llcr(cfads_loan_life, out_debt, interest_rate), 3)
            
            # Remaining CFADS over project life (concession period)
            cfads_project_life = [
                r["cfads"] for r in waterfall_records 
                if current_yr <= r["year"] <= concession_period and r["phase"] == "operations"
            ]
            rec["plcr"] = round(self.calculate_plcr(cfads_project_life, out_debt, interest_rate), 3)
            
        return waterfall_records

    def compute_project_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processes a projects DataFrame, runs SPV waterfall simulations,
        and computes Feast-aligned financial features.
        """
        features_list = []
        
        for _, row in df.iterrows():
            proj_id = row["project_id"]
            inv_val = row["investment_value_usd_m"]
            debt_val = row["debt_value_usd_m"]
            concession_period = int(row["concession_period_years"])
            base_dscr = row["dscr"]
            sector = row["sector"]
            
            # Run simulation
            waterfall = self.simulate_spv_waterfall(
                investment_value=inv_val,
                debt_value=debt_val,
                concession_period=concession_period,
                base_dscr=base_dscr,
                sector=sector
            )
            
            # Extract features from waterfall
            op_records = [r for r in waterfall if r["phase"] == "operations"]
            
            if op_records:
                avg_dscr = np.mean([r["dscr"] for r in op_records])
                min_dscr = np.min([r["dscr"] for r in op_records])
                avg_llcr = np.mean([r["llcr"] for r in op_records])
                min_llcr = np.min([r["llcr"] for r in op_records])
                avg_plcr = np.mean([r["plcr"] for r in op_records])
                min_plcr = np.min([r["plcr"] for r in op_records])
                total_equity_dist = sum([r["equity_distribution"] for r in op_records])
                total_prepayments = sum([r["prepayment"] for r in op_records])
            else:
                avg_dscr = min_dscr = avg_llcr = min_llcr = avg_plcr = min_plcr = 0.0
                total_equity_dist = total_prepayments = 0.0
                
            features_list.append({
                "project_id": proj_id,
                "leverage_ratio": round(debt_val / inv_val if inv_val > 0 else 0, 4),
                "debt_to_equity": round(row["debt_equity_ratio"], 4),
                "debt_tenor_years": concession_period - 8,  # 3 construction + 5 tail
                "simulated_avg_dscr": round(avg_dscr, 3),
                "simulated_min_dscr": round(min_dscr, 3),
                "simulated_avg_llcr": round(avg_llcr, 3),
                "simulated_min_llcr": round(min_llcr, 3),
                "simulated_avg_plcr": round(avg_plcr, 3),
                "simulated_min_plcr": round(min_plcr, 3),
                "total_equity_distributions_usd_m": round(total_equity_dist, 3),
                "total_debt_prepayments_usd_m": round(total_prepayments, 3),
                "event_timestamp": pd.Timestamp.now()  # Feast alignment
            })
            
        features_df = pd.DataFrame(features_list)
        return features_df

# Example definition of a Feast Feature View configuration structure
FEAST_METADATA = {
    "entity": "project_id",
    "features": [
        "leverage_ratio",
        "debt_to_equity",
        "debt_tenor_years",
        "simulated_avg_dscr",
        "simulated_min_dscr",
        "simulated_avg_llcr",
        "simulated_min_llcr",
        "simulated_avg_plcr",
        "simulated_min_plcr",
        "total_equity_distributions_usd_m",
        "total_debt_prepayments_usd_m"
    ]
}
