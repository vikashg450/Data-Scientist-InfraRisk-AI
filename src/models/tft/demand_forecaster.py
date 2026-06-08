import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Tuple, Optional

# Gated Linear Unit
class GLU(nn.Module):
    def __init__(self, d_model: int):
        super(GLU, self).__init__()
        self.fc = nn.Linear(d_model, d_model * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fc = self.fc(x)
        x_sig, x_gate = torch.chunk(x_fc, 2, dim=-1)
        return x_sig * torch.sigmoid(x_gate)

# Gated Residual Network (GRN)
class GRN(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1, context_dim: Optional[int] = None):
        super(GRN, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, d_hidden, bias=False)
        else:
            self.context_fc = None
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.glu = GLU(d_hidden)
        self.project = nn.Linear(d_hidden, d_out) if d_hidden != d_out else nn.Identity()
        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_out)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        shortcut = self.shortcut(x)
        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            # Broadcast context if it has fewer dimensions
            if context.dim() == 2 and h.dim() == 3:
                context_expanded = context.unsqueeze(1).expand(-1, h.size(1), -1)
                h = h + self.context_fc(context_expanded)
            else:
                h = h + self.context_fc(context)
        h = torch.relu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        h = self.glu(h)
        h = self.project(h)
        return self.layernorm(h + shortcut)

# Temporal Fusion Transformer Core Model
class DemandTFT(nn.Module):
    def __init__(self, num_static: int, num_past: int, num_future: int, d_model: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super(DemandTFT, self).__init__()
        self.num_static = num_static
        self.num_past = num_past
        self.num_future = num_future
        self.d_model = d_model
        
        # Input Variable Selection / Encoding using GRN
        self.static_grn = GRN(num_static, d_model, d_model, dropout=dropout)
        self.past_grn = GRN(num_past, d_model, d_model, dropout=dropout)
        self.future_grn = GRN(num_future, d_model, d_model, dropout=dropout)
        
        # Temporal Processing (LSTMs)
        self.lstm_past = nn.LSTM(d_model, d_model, batch_first=True)
        self.lstm_future = nn.LSTM(d_model, d_model, batch_first=True)
        
        # Temporal Self-Attention
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=dropout)
        self.post_attn_grn = GRN(d_model, d_model, d_model, dropout=dropout)
        
        # Quantile Output Head (P10, P50, P90 outputs)
        self.output_fc = nn.Linear(d_model, 3)
        
    def forward(self, x_static: torch.Tensor, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        """
        x_static: (B, num_static)
        x_past: (B, T_past, num_past)
        x_future: (B, T_future, num_future)
        Returns: (B, T_future, 3) where the last dim contains P10, P50, P90 forecasts.
        """
        # Encode static covariates
        static_ctx = self.static_grn(x_static)  # (B, d_model)
        
        # Encode past features
        B, T_past, _ = x_past.shape
        past_encoded = self.past_grn(x_past, context=static_ctx)  # (B, T_past, d_model)
        
        # Encode future features
        B, T_future, _ = x_future.shape
        future_encoded = self.future_grn(x_future, context=static_ctx)  # (B, T_future, d_model)
        
        # LSTM sequence learning
        past_lstm, (h_n, c_n) = self.lstm_past(past_encoded)
        future_lstm, _ = self.lstm_future(future_encoded, (h_n, c_n))
        
        # Attention over temporal representations
        attn_out, _ = self.attn(query=future_lstm, key=past_lstm, value=past_lstm)
        
        # Post-attention gating
        out = self.post_attn_grn(attn_out + future_lstm)
        
        # Quantile prediction
        quantiles = self.output_fc(out)  # (B, T_future, 3)
        return quantiles

def compute_quantile_loss(y_pred: torch.Tensor, y_true: torch.Tensor, quantiles: List[float] = [0.1, 0.5, 0.9]) -> torch.Tensor:
    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(-1)
    
    losses = []
    for i, q in enumerate(quantiles):
        error = y_true - y_pred[..., i:i+1]
        loss = torch.max((q - 1) * error, q * error)
        losses.append(loss.mean())
    return torch.stack(losses).sum()

class TFTForecaster:
    """Wrapper class for training and predicting with the DemandTFT model."""
    def __init__(self, num_static: int, num_past: int, num_future: int, d_model: int = 64, num_heads: int = 4, lr: float = 1e-3):
        self.model = DemandTFT(num_static, num_past, num_future, d_model, num_heads)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def fit(self, x_static: np.ndarray, x_past: np.ndarray, x_future: np.ndarray, y_future: np.ndarray, epochs: int = 10, batch_size: int = 32):
        self.model.train()
        dataset_size = len(x_static)
        
        x_static_t = torch.tensor(x_static, dtype=torch.float32)
        x_past_t = torch.tensor(x_past, dtype=torch.float32)
        x_future_t = torch.tensor(x_future, dtype=torch.float32)
        y_future_t = torch.tensor(y_future, dtype=torch.float32)
        
        for epoch in range(epochs):
            permutation = torch.randperm(dataset_size)
            epoch_loss = 0.0
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                
                self.optimizer.zero_grad()
                pred = self.model(x_static_t[indices], x_past_t[indices], x_future_t[indices])
                loss = compute_quantile_loss(pred, y_future_t[indices])
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(indices)
                
            # print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss/dataset_size:.4f}")
            
    def predict(self, x_static: np.ndarray, x_past: np.ndarray, x_future: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts quantiles P10, P50, P90.
        Returns:
            p10: (B, T_future)
            p50: (B, T_future)
            p90: (B, T_future)
        """
        self.model.eval()
        with torch.no_grad():
            x_static_t = torch.tensor(x_static, dtype=torch.float32)
            x_past_t = torch.tensor(x_past, dtype=torch.float32)
            x_future_t = torch.tensor(x_future, dtype=torch.float32)
            
            pred = self.model(x_static_t, x_past_t, x_future_t).numpy()
            
            p10 = pred[..., 0]
            p50 = pred[..., 1]
            p90 = pred[..., 2]
            return p10, p50, p90

# SARIMA Baseline Model
class SARIMABaseline:
    """SARIMA baseline demand forecasting model using statsmodels."""
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1), seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None
        
    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        # Fill missing values and run model fit
        y_clean = y.ffill().bfill()
        if exog is not None:
            exog_clean = exog.ffill().bfill()
        else:
            exog_clean = None
            
        model = SARIMAX(
            y_clean, 
            exog=exog_clean, 
            order=self.order, 
            seasonal_order=self.seasonal_order, 
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        self.model_fit = model.fit(disp=False)
        
    def predict(self, steps: int, exog: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns predictions for the next `steps` periods.
        Returns:
            p10, p50, p90 arrays of size (steps,)
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted first.")
            
        if exog is not None:
            exog_clean = exog.ffill().bfill()
        else:
            exog_clean = None
            
        forecast_res = self.model_fit.get_forecast(steps=steps, exog=exog_clean)
        p50 = forecast_res.predicted_mean.values
        se = forecast_res.se_mean.values
        
        # Construct standard quantiles based on normality of forecast errors
        # P10 is approx 1.28 standard deviations below mean
        # P90 is approx 1.28 standard deviations above mean
        p10 = p50 - 1.28 * se
        p90 = p50 + 1.28 * se
        
        return p10, p50, p90


def calculate_picp(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> float:
    """Computes Prediction Interval Coverage Probability (PICP)."""
    covered = (y_true >= p10) & (y_true <= p90)
    return float(np.mean(covered) * 100)

def calculate_crps(y_true: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> float:
    """Computes an approximation of Continuous Ranked Probability Score (CRPS) using pinball loss."""
    pinball = 0.0
    for q, pred in zip([0.1, 0.5, 0.9], [p10, p50, p90]):
        err = y_true - pred
        loss = np.maximum((q - 1.0) * err, q * err)
        pinball += np.mean(loss)
    return pinball / 3.0

def calculate_winkler_score(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray, alpha: float = 0.10) -> float:
    """Computes the Winkler Score for prediction intervals."""
    scores = []
    for true, l, u in zip(y_true, p10, p90):
        width = u - l
        if true < l:
            score = width + (2.0 / alpha) * (l - true)
        elif true > u:
            score = width + (2.0 / alpha) * (true - u)
        else:
            score = width
        scores.append(score)
    return float(np.mean(scores))


class TollRoadDemandModel:
    """Toll road traffic demand module with logistic ramp-up curve."""
    @staticmethod
    def calculate_traffic(t: np.ndarray, mature_adt: float, k: float, t0: float) -> np.ndarray:
        # Traffic(t) = Mature_ADT / (1 + e^(-k*(t-t0)))
        return mature_adt / (1.0 + np.exp(-k * (t - t0)))


class PowerPlantDemandModel:
    """Power plant generation demand module (capacity + energy payment model)."""
    @staticmethod
    def calculate_revenue(capacity_mw: float, capacity_tariff_usd_mw_yr: float, 
                          generation_mwh: float, energy_tariff_usd_mwh: float, 
                          availability_pct: float = 0.95) -> float:
        capacity_payment = capacity_mw * capacity_tariff_usd_mw_yr * availability_pct
        energy_payment = generation_mwh * energy_tariff_usd_mwh
        return capacity_payment + energy_payment


class PortDemandModel:
    """Port throughput demand module (macro trade flow + port choice model)."""
    @staticmethod
    def calculate_throughput(gdp_growth_pct: float, baseline_throughput: float, 
                             elasticity: float = 1.5, port_attractiveness_score: float = 0.6) -> float:
        trade_growth = gdp_growth_pct * elasticity
        total_market_throughput = baseline_throughput * (1.0 + trade_growth)
        market_share = 1.0 / (1.0 + np.exp(-5.0 * (port_attractiveness_score - 0.5)))
        return total_market_throughput * market_share


class AirportDemandModel:
    """Airport passenger demand module (domestic, international, and transfer separately)."""
    @staticmethod
    def calculate_passengers(domestic_baseline: float, international_baseline: float, transfer_baseline: float,
                             gdp_growth: float, fx_depreciation: float, transfer_incentive: float = 0.05) -> Dict[str, float]:
        dom_growth = gdp_growth * 1.1
        domestic = domestic_baseline * (1.0 + dom_growth)
        
        intl_growth = gdp_growth * 0.8 - fx_depreciation * 0.4
        international = international_baseline * (1.0 + intl_growth)
        
        transfer_growth = 0.02 + transfer_incentive
        transfer = transfer_baseline * (1.0 + transfer_growth)
        
        return {
            "domestic": max(0.0, domestic),
            "international": max(0.0, international),
            "transfer": max(0.0, transfer),
            "total": max(0.0, domestic + international + transfer)
        }


class SectorDemandForecaster:
    """
    Manager for sector-specific demand forecasting models (toll roads, power, ports)
    wrapping the TFTForecaster and SARIMABaseline models.
    """
    def __init__(self, sector: str, num_static: int, num_past: int, num_future: int):
        self.sector = sector.lower().replace(" ", "_")
        self.tft_model = TFTForecaster(num_static=num_static, num_past=num_past, num_future=num_future)
        self.sarima_model = SARIMABaseline()
        
    def fit(self, x_static: np.ndarray, x_past: np.ndarray, x_future: np.ndarray, y_future: np.ndarray, y_series: pd.Series):
        """Fits both TFT and SARIMA models for the sector."""
        self.tft_model.fit(x_static, x_past, x_future, y_future, epochs=5, batch_size=16)
        self.sarima_model.fit(y_series)
        
    def predict(self, x_static: np.ndarray, x_past: np.ndarray, x_future: np.ndarray, steps: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Runs predictions for both the TFT and SARIMA baseline models.
        """
        tft_p10, tft_p50, tft_p90 = self.tft_model.predict(x_static, x_past, x_future)
        sarima_p10, sarima_p50, sarima_p90 = self.sarima_model.predict(steps=steps)
        return {
            "TFT": (tft_p10, tft_p50, tft_p90),
            "SARIMA": (sarima_p10, sarima_p50, sarima_p90)
        }

    def backtest_models(self, time_series: pd.Series, train_split_year: int = 2017) -> Dict[str, Any]:
        """
        Backtests the SARIMA forecaster via a rolling window.
        """
        train_data = time_series[time_series.index.year <= train_split_year]
        test_data = time_series[(time_series.index.year > train_split_year) & (time_series.index.year <= 2022)]
        
        if len(train_data) == 0 or len(test_data) == 0:
            mid = len(time_series) * 3 // 4
            train_data = time_series.iloc[:mid]
            test_data = time_series.iloc[mid:]
            
        self.sarima_model.fit(train_data)
        p10, p50, p90 = self.sarima_model.predict(steps=len(test_data))
        
        y_true = test_data.values
        mape = float(np.mean(np.abs(y_true - p50) / np.maximum(y_true, 1e-5)) * 100)
        picp = calculate_picp(y_true, p10, p90)
        crps = calculate_crps(y_true, p10, p50, p90)
        winkler = calculate_winkler_score(y_true, p10, p90)
        
        return {
            "mape": round(mape, 3),
            "picp": round(picp, 1),
            "crps": round(crps, 3),
            "winkler": round(winkler, 3),
            "pi_coverage_valid": picp >= 90.0
        }

