import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MarketDataLoader:
    """
    Downloads and caches market data (commodities, FX rates, interest rate curves)
    using yfinance, and provides simulated data generators as a fallback.
    """

    COMMODITIES = {
        "Crude_Oil": "CL=F",      # WTI Crude
        "Natural_Gas": "NG=F",    # Henry Hub Natural Gas
        "Steel": "HRC=F",         # Hot Rolled Coil Steel Futures
        "Cement_Proxy": "CX",     # Cemex stock as a proxy for Cement price trends
    }

    FX_RATES = {
        "USD_EUR": "EURUSD=X",
        "USD_GBP": "GBPUSD=X",
        "USD_INR": "INR=X",
        "USD_BRL": "BRL=X",
        "USD_ZAF": "ZAR=X",
        "USD_TRY": "TRY=X",
        "USD_EGY": "EGP=X",
    }

    # US Treasuries for SOFR curve proxying
    US_YIELDS = {
        "3M": "^IRX",
        "5Y": "^FVX",
        "10Y": "^TNX",
        "30Y": "^TYX"
    }

    def __init__(self, cache_dir: str = "data/market"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def download_ticker_history(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Helper method to download a single ticker's history from yfinance.
        """
        try:
            logger.info(f"Downloading history for {ticker} from {start_date} to {end_date}...")
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not df.empty:
                # Format index and columns
                df = df[['Close']].rename(columns={'Close': ticker})
                return df
        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")
        return pd.DataFrame()

    def fetch_all_market_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetches all market data from yfinance. If any fail or if offline,
        returns simulated market data.
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=3650)).strftime("%Y-%m-%d") # 10 years
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Attempting real market data download from {start_date} to {end_date}...")

        # Collect dataframes
        dfs = []
        
        # Test connection by trying one ticker first
        test_df = self.download_ticker_history("CL=F", (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"), end_date)
        if test_df.empty:
            logger.warning("Could not establish connection to yfinance. Falling back to synthetic market data generator.")
            return self.generate_synthetic_market_data(start_date, end_date)

        # Download Commodities
        for name, ticker in self.COMMODITIES.items():
            df = self.download_ticker_history(ticker, start_date, end_date)
            if not df.empty:
                df.columns = [name]
                dfs.append(df)
            else:
                logger.warning(f"Using synthetic fallback for commodity {name}")
                dfs.append(self.generate_single_synthetic_series(name, 100.0, 0.05, 0.20, start_date, end_date))

        # Download FX Rates
        for name, ticker in self.FX_RATES.items():
            df = self.download_ticker_history(ticker, start_date, end_date)
            if not df.empty:
                df.columns = [name]
                dfs.append(df)
            else:
                logger.warning(f"Using synthetic fallback for FX rate {name}")
                dfs.append(self.generate_single_synthetic_series(name, 1.0 if "EUR" in name or "GBP" in name else 10.0, 0.02, 0.12, start_date, end_date))

        # Download Yields
        for name, ticker in self.US_YIELDS.items():
            df = self.download_ticker_history(ticker, start_date, end_date)
            if not df.empty:
                # yfinance yield tickers return values scaled by 10 (e.g. 4.5% is 45.0)
                # Let's scale back to standard percentage values
                df = df / 10.0
                df.columns = [f"US_Yield_{name}"]
                dfs.append(df)
            else:
                logger.warning(f"Using synthetic fallback for yield {name}")
                dfs.append(self.generate_single_synthetic_series(f"US_Yield_{name}", 3.0, 0.01, 0.05, start_date, end_date))

        # Merge all data on date index
        if dfs:
            merged_df = dfs[0]
            for df in dfs[1:]:
                merged_df = merged_df.join(df, how="outer")
            
            # Forward fill/backward fill to handle calendar day mismatches (futures vs currency vs stocks)
            merged_df = merged_df.ffill().bfill()
            
            # Fit and add Nelson-Siegel Curve parameters for SOFR
            merged_df = self.add_nelson_siegel_parameters(merged_df)
            
            # Generate simulated CDS spreads (OTC data, must be simulated)
            merged_df = self.add_synthetic_cds_spreads(merged_df)
            
            # Cache to CSV
            output_path = os.path.join(self.cache_dir, "market_data_combined.csv")
            merged_df.to_csv(output_path)
            logger.info(f"Market data successfully processed and cached to {output_path}")
            return merged_df

        return self.generate_synthetic_market_data(start_date, end_date)

    def generate_single_synthetic_series(self, name: str, start_val: float, drift: float, vol: float, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generates a single geometric Brownian motion time series.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        n = len(dates)
        dt = 1/252
        
        # Simulating returns
        np.random.seed(hash(name) % 2**32)
        rand_norms = np.random.normal(0, 1, n)
        returns = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * rand_norms
        price_path = start_val * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame(price_path, index=dates, columns=[name])
        df.index.name = "Date"
        return df

    def generate_synthetic_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generates a full mock dataset of commodities, FX rates, interest rates,
        Nelson-Siegel parameters, and CDS spreads.
        """
        logger.info("Generating fully synthetic market data...")
        dates = pd.date_range(start=start_date, end=end_date, freq="B")
        
        # Commodities
        df_oil = self.generate_single_synthetic_series("Crude_Oil", 75.0, 0.02, 0.25, start_date, end_date)
        df_gas = self.generate_single_synthetic_series("Natural_Gas", 3.0, 0.05, 0.35, start_date, end_date)
        df_steel = self.generate_single_synthetic_series("Steel", 800.0, -0.01, 0.20, start_date, end_date)
        df_cement = self.generate_single_synthetic_series("Cement_Proxy", 8.0, 0.03, 0.15, start_date, end_date)
        
        # FX Rates
        df_eur = self.generate_single_synthetic_series("USD_EUR", 1.10, 0.0, 0.08, start_date, end_date)
        df_gbp = self.generate_single_synthetic_series("USD_GBP", 1.25, 0.0, 0.09, start_date, end_date)
        df_inr = self.generate_single_synthetic_series("USD_INR", 80.0, 0.02, 0.05, start_date, end_date)
        df_brl = self.generate_single_synthetic_series("USD_BRL", 5.0, 0.04, 0.15, start_date, end_date)
        df_zar = self.generate_single_synthetic_series("USD_ZAF", 18.0, 0.03, 0.18, start_date, end_date)
        df_try = self.generate_single_synthetic_series("USD_TRY", 20.0, 0.25, 0.22, start_date, end_date)
        df_egy = self.generate_single_synthetic_series("USD_EGY", 30.0, 0.15, 0.12, start_date, end_date)
        
        # Yields
        df_y3m = self.generate_single_synthetic_series("US_Yield_3M", 4.5, 0.0, 0.08, start_date, end_date)
        df_y5y = self.generate_single_synthetic_series("US_Yield_5Y", 4.0, 0.0, 0.06, start_date, end_date)
        df_y10y = self.generate_single_synthetic_series("US_Yield_10Y", 4.2, 0.0, 0.05, start_date, end_date)
        df_y30y = self.generate_single_synthetic_series("US_Yield_30Y", 4.3, 0.0, 0.04, start_date, end_date)

        # Merge
        df = df_oil.join([
            df_gas, df_steel, df_cement,
            df_eur, df_gbp, df_inr, df_brl, df_zar, df_try, df_egy,
            df_y3m, df_y5y, df_y10y, df_y30y
        ], how="outer").ffill().bfill()
        
        # Add Nelson-Siegel and CDS
        df = self.add_nelson_siegel_parameters(df)
        df = self.add_synthetic_cds_spreads(df)
        
        output_path = os.path.join(self.cache_dir, "market_data_combined.csv")
        df.to_csv(output_path)
        logger.info(f"Fully synthetic market data saved to {output_path}")
        return df

    def add_nelson_siegel_parameters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fits or generates Nelson-Siegel Curve parameters (beta0, beta1, beta2, tau) 
        representing the interest rate yield curves for SOFR and EURIBOR.
        """
        n = len(df)
        np.random.seed(101)
        
        # Fit Nelson-Siegel for SOFR (represented by US Yields)
        # Yield(t) = beta0 + beta1*((1-e^-t/tau)/(t/tau)) + beta2*((1-e^-t/tau)/(t/tau) - e^-t/tau)
        # We can simulate parameters that trend with the downloaded yields
        # beta0: long term rate level (approx US_Yield_30Y)
        # beta1: short term rate level relative to long term (approx US_Yield_3M - US_Yield_30Y)
        # beta2: medium term hump (approx 2 * US_Yield_5Y - US_Yield_3M - US_Yield_30Y)
        # tau: decay parameter, usually constant ~ 1.5 to 2.5
        
        y_3m = df["US_Yield_3M"] if "US_Yield_3M" in df.columns else np.full(n, 4.0)
        y_5y = df["US_Yield_5Y"] if "US_Yield_5Y" in df.columns else np.full(n, 3.8)
        y_30y = df["US_Yield_30Y"] if "US_Yield_30Y" in df.columns else np.full(n, 4.2)
        
        df["sofr_ns_beta0"] = y_30y + np.random.normal(0, 0.05, n)
        df["sofr_ns_beta1"] = y_3m - y_30y + np.random.normal(0, 0.05, n)
        df["sofr_ns_beta2"] = 2.0 * y_5y - y_3m - y_30y + np.random.normal(0, 0.1, n)
        df["sofr_ns_tau"] = 2.0 + np.random.normal(0, 0.02, n)
        
        # Generate EURIBOR parameters (slightly lower rates, steeper curve)
        df["euribor_ns_beta0"] = y_30y - 1.0 + np.random.normal(0, 0.05, n)
        df["euribor_ns_beta1"] = (y_3m - 1.5) - (y_30y - 1.0) + np.random.normal(0, 0.05, n)
        df["euribor_ns_beta2"] = 1.5 * y_5y - y_3m - y_30y + np.random.normal(0, 0.1, n)
        df["euribor_ns_tau"] = 1.8 + np.random.normal(0, 0.02, n)
        
        return df

    def add_synthetic_cds_spreads(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates realistic Credit Default Swap (CDS) spreads (in bps) 
        for different sovereign risk ratings (AAA to CCC).
        """
        n = len(df)
        np.random.seed(202)
        
        # Define CDS spread base levels by rating
        # Rating -> Base spread in bps
        ratings_base = {
            "AAA": (15, 5),
            "AA": (30, 8),
            "A": (60, 15),
            "BBB": (120, 25),
            "BB": (280, 60),
            "B": (550, 110),
            "CCC": (1200, 250)
        }
        
        # Generate daily walks for each rating bracket
        for rating, (base, vol) in ratings_base.items():
            rand_norms = np.random.normal(0, 1, n)
            walk = np.cumsum(rand_norms * (vol / np.sqrt(252)))
            spread = base * np.exp(walk)
            df[f"CDS_Spread_{rating}"] = np.round(spread, 1)
            
        return df

if __name__ == "__main__":
    loader = MarketDataLoader()
    df = loader.fetch_all_market_data()
    print("Market Data Shape:", df.shape)
    print(df.tail())
