import os
import logging
import numpy as np
import pandas as pd
import rasterio
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class SatelliteFeaturesExtractor:
    """
    Extracts NDVI, NDBI, vegetation clearance ratio, and temporal construction
    progress curves from Sentinel-2 satellite imagery.
    """

    def __init__(self, data_dir: str = "data/satellite"):
        self.data_dir = data_dir

    def load_raster_bands(self, tiff_path: str) -> Dict[str, np.ndarray]:
        """
        Loads the relevant spectral bands from the Sentinel-2 GeoTIFF.
        Sentinel-2 Band mapping:
        - Band 4 (Red): index 4
        - Band 8 (NIR): index 8
        - Band 12 (SWIR1/B11): index 12
        """
        if not os.path.exists(tiff_path):
            raise FileNotFoundError(f"Satellite TIFF file not found: {tiff_path}")

        bands = {}
        with rasterio.open(tiff_path) as src:
            # Band 4: Red
            bands["B4"] = src.read(4).astype(np.float32)
            # Band 8: NIR
            bands["B8"] = src.read(8).astype(np.float32)
            # Band 12: SWIR1 (B11 in Sentinel-2)
            bands["B12"] = src.read(12).astype(np.float32)
            
        return bands

    @staticmethod
    def calculate_ndvi(bands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Computes Normalized Difference Vegetation Index (NDVI).
        Formula: NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4)
        """
        nir = bands["B8"]
        red = bands["B4"]
        denominator = nir + red
        # Avoid division by zero
        denominator[denominator == 0.0] = 1e-5
        return (nir - red) / denominator

    @staticmethod
    def calculate_ndbi(bands: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Computes Normalized Difference Built-Up Index (NDBI).
        Formula: NDBI = (SWIR1 - NIR) / (SWIR1 + NIR) = (B12 - B8) / (B12 + B8)
        """
        swir1 = bands["B12"]
        nir = bands["B8"]
        denominator = swir1 + nir
        # Avoid division by zero
        denominator[denominator == 0.0] = 1e-5
        return (swir1 - nir) / denominator

    def extract_raster_features(self, tiff_path: str) -> Dict[str, float]:
        """
        Computes mean NDVI and NDBI over the project site image.
        """
        try:
            bands = self.load_raster_bands(tiff_path)
            ndvi = self.calculate_ndvi(bands)
            ndbi = self.calculate_ndbi(bands)
            
            # Mask out invalid values/pixels (e.g. padding/zero values if any)
            mask = (bands["B4"] > 0) & (bands["B8"] > 0)
            if not mask.any():
                return {"mean_ndvi": 0.0, "mean_ndbi": 0.0}
                
            return {
                "mean_ndvi": float(np.mean(ndvi[mask])),
                "mean_ndbi": float(np.mean(ndbi[mask]))
            }
        except Exception as e:
            logger.error(f"Error extracting raster features from {tiff_path}: {e}")
            return {"mean_ndvi": 0.0, "mean_ndbi": 0.0}

    def compute_progress_curve(self, elapsed_months: int, planned_duration_months: int) -> float:
        """
        Simulates an S-curve for planned construction progress.
        Formula: Progress(t) = 1 / (1 + exp(-k * (t - t0)))
        Normalised to be 0 at t=0 and 1 at t=planned_duration_months.
        """
        if elapsed_months <= 0:
            return 0.0
        if elapsed_months >= planned_duration_months:
            return 1.0
            
        t0 = planned_duration_months / 2.0
        k = 10.0 / planned_duration_months  # Steepness factor
        
        raw_val = 1.0 / (1.0 + np.exp(-k * (elapsed_months - t0)))
        raw_min = 1.0 / (1.0 + np.exp(k * t0))
        raw_max = 1.0 / (1.0 + np.exp(-k * (planned_duration_months - t0)))
        
        normalized_progress = (raw_val - raw_min) / (raw_max - raw_min)
        return float(np.clip(normalized_progress, 0.0, 1.0))

    def extract_project_satellite_features(
        self,
        project_id: str,
        elapsed_months: int,
        planned_duration_months: int,
        baseline_ndvi: float = 0.35,
        baseline_ndbi: float = -0.15,
        target_ndbi: float = 0.42
    ) -> Dict[str, Any]:
        """
        Computes temporal satellite progress features by comparing current
        imagery with baseline construction assumptions.
        """
        tiff_path = os.path.join(self.data_dir, f"{project_id}_sentinel2.tif")
        
        # Load indices
        if os.path.exists(tiff_path):
            indices = self.extract_raster_features(tiff_path)
            mean_ndvi = indices["mean_ndvi"]
            mean_ndbi = indices["mean_ndbi"]
        else:
            # Offline/mock fallback if TIFF doesn't exist
            # Simulate features based on elapsed months (roughly matches progress)
            progress_ratio = elapsed_months / planned_duration_months if planned_duration_months > 0 else 1.0
            progress_ratio = np.clip(progress_ratio, 0.0, 1.0)
            
            mean_ndvi = baseline_ndvi - (baseline_ndvi - 0.08) * progress_ratio
            mean_ndbi = baseline_ndbi + (target_ndbi - baseline_ndbi) * progress_ratio
            logger.warning(f"TIFF not found for {project_id}, using simulated mock values.")

        # 1. Vegetation Clearance Ratio
        # (baseline_ndvi - current_ndvi) / baseline_ndvi
        if baseline_ndvi > 0:
            clearance_ratio = (baseline_ndvi - mean_ndvi) / baseline_ndvi
        else:
            clearance_ratio = 0.0
        clearance_ratio = float(np.clip(clearance_ratio, 0.0, 1.0))

        # 2. NDBI Progression Rate
        # (current_ndbi - baseline_ndbi) / elapsed_months
        if elapsed_months > 0:
            ndbi_progression_rate = (mean_ndbi - baseline_ndbi) / elapsed_months
        else:
            ndbi_progression_rate = 0.0

        # 3. Satellite Progress Estimate (Fusion of clearance and built-up index)
        # Early phase: site preparation (NDVI clearance).
        # Later phase: structure buildup (NDBI increase).
        if clearance_ratio < 0.90:
            satellite_progress = clearance_ratio * 0.30
        else:
            ndbi_span = target_ndbi - baseline_ndbi
            if ndbi_span > 0:
                ndbi_ratio = (mean_ndbi - baseline_ndbi) / ndbi_span
            else:
                ndbi_ratio = 1.0
            satellite_progress = 0.30 + float(np.clip(ndbi_ratio, 0.0, 1.0)) * 0.70
            
        satellite_progress = float(np.clip(satellite_progress, 0.0, 1.0))

        # 4. Planned Progress S-Curve
        planned_progress = self.compute_progress_curve(elapsed_months, planned_duration_months)

        # 5. Progress Deviation and Delay Estimation
        progress_deviation = satellite_progress - planned_progress
        
        # Estimate delay in months based on deviation
        if progress_deviation < 0:
            # Behind schedule
            schedule_delay_months = float(np.abs(progress_deviation) * planned_duration_months)
        else:
            schedule_delay_months = 0.0

        return {
            "project_id": project_id,
            "mean_ndvi": round(mean_ndvi, 4),
            "mean_ndbi": round(mean_ndbi, 4),
            "vegetation_clearance_ratio": round(clearance_ratio, 4),
            "ndbi_progression_rate": round(ndbi_progression_rate, 4),
            "satellite_progress_estimate": round(satellite_progress, 4),
            "planned_progress_curve": round(planned_progress, 4),
            "progress_deviation": round(progress_deviation, 4),
            "schedule_delay_months": round(schedule_delay_months, 2),
            "event_timestamp": pd.Timestamp.now()
        }

    def compute_all_satellite_features(self, projects_df: pd.DataFrame, current_elapsed_months: int = 12) -> pd.DataFrame:
        """
        Extracts satellite features for a whole DataFrame of projects.
        """
        features_list = []
        for _, row in projects_df.iterrows():
            proj_id = row["project_id"]
            # Assume 36 months construction period if not specified
            planned_duration = 36
            
            features = self.extract_project_satellite_features(
                project_id=proj_id,
                elapsed_months=current_elapsed_months,
                planned_duration_months=planned_duration
            )
            features_list.append(features)
            
        return pd.DataFrame(features_list)

FEAST_METADATA = {
    "entity": "project_id",
    "features": [
        "mean_ndvi",
        "mean_ndbi",
        "vegetation_clearance_ratio",
        "ndbi_progression_rate",
        "satellite_progress_estimate",
        "planned_progress_curve",
        "progress_deviation",
        "schedule_delay_months"
    ]
}
