import numpy as np
import scipy.stats
from typing import List, Dict, Any
from .base_metric import BaseMetric

class Aggregator:
    """Aggregates metrics across multiple backtest runs"""
    
    @staticmethod
    def aggregate(results: List[Dict]) -> Dict[str, Dict]:
        """Aggregate metrics from multiple backtest results"""
        aggregated_results = {}
        
        if not results:
            return aggregated_results

        for metric in results[0].keys():
            if metric == "performance_data":
                continue

            values = [result[metric] for result in results]
            values_array = np.array(values)
            
            if isinstance(values[0], bool):
                aggregated_results[metric] = Aggregator._aggregate_boolean(values_array)
            else:
                aggregated_results[metric] = Aggregator._aggregate_numeric(values_array)
                
        return aggregated_results

    @staticmethod
    def _aggregate_boolean(values: np.ndarray) -> Dict[str, Any]:
        """Aggregate boolean metrics"""
        true_count = sum(values)
        total_count = len(values)
        return {
            "mean": true_count / total_count if total_count > 0 else 0,
            "count": total_count,
            "true_ratio": true_count / total_count if total_count > 0 else 0,
        }

    @staticmethod
    def _aggregate_numeric(values: np.ndarray) -> Dict[str, Any]:
        """Aggregate numeric metrics"""
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            return Aggregator._empty_numeric_stats(len(values))
            
        return {
            "mean": np.nanmean(values),
            "median": np.nanmedian(values),
            "std": np.nanstd(values),
            "min": np.nanmin(values),
            "max": np.nanmax(values),
            "percentile_25": np.percentile(valid_values, 25),
            "percentile_75": np.percentile(valid_values, 75),
            "percentile_90": np.percentile(valid_values, 90),
            "percentile_95": np.percentile(valid_values, 95),
            "skewness": scipy.stats.skew(valid_values),
            "kurtosis": scipy.stats.kurtosis(valid_values),
            "count": len(valid_values),
            "iqr": np.percentile(valid_values, 75) - np.percentile(valid_values, 25),
        }

    @staticmethod
    def _empty_numeric_stats(count: int) -> Dict[str, Any]:
        """Return empty stats structure for invalid numeric metrics"""
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "percentile_25": np.nan,
            "percentile_75": np.nan,
            "percentile_90": np.nan,
            "percentile_95": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "count": count,
            "iqr": np.nan,
        }
