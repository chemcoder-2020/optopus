import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from optopus.trades.external_entry_conditions import (
    CompositePipelineCondition,
    RSIComponent,
    AndComponent,
    OrComponent,
    NotComponent
)
from optopus.utils.ohlc_data_processor import DataProcessor

class MockStrategy:
    def __init__(self, underlying_price):
        self.underlying_last = underlying_price
        
class MockManager:
    def __init__(self):
        self.context = {}

def test_composite_pipeline_operators():
    # Create mock historical data with downward trend (oversold condition)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    closes = np.concatenate([np.linspace(100, 85, 90), np.linspace(85, 80, 10)])
    hist_data = pd.DataFrame({
        "close": closes,
        "high": closes + 2,
        "low": closes - 2,
        "open": closes
    }, index=dates)
    
    # Create test components
    rsi_oversold = RSIComponent(period=14, oversold=30)
    linear_reg = AndComponent.left(
        "LinearRegression", {"lag": 14}
    )
    pipeline = (
        rsi_oversold * 
        linear_reg *
        OrComponent.left(
            "MedianTrend", 
            {"short_lag": 50, "long_lag": 200}
        )
    )
    
    # Create composite condition
    processor = DataProcessor(hist_data)
    condition = CompositePipelineCondition(
        pipeline=pipeline,
        ohlc_data=hist_data
    )
    
    strategy = MockStrategy(underlying_price=80)
    manager = MockManager()
    current_time = datetime(2023, 4, 1)
    
    # Test valid entry condition
    manager.context["historical_data"] = hist_data
    assert condition.should_enter(
        time=current_time, 
        strategy=strategy, 
        manager=manager
    ), "Should enter when all conditions met"
    
    # Test failed RSI condition
    strict_rsi = RSIComponent(period=14, oversold=50)
    strict_pipeline = strict_rsi * pipeline
    condition.pipeline = strict_pipeline
    assert not condition.should_enter(
        time=current_time,
        strategy=strategy,
        manager=manager
    ), "Should not enter when RSI too high"
    
    # Test nested logic
    complex_pipeline = (
        (rsi_oversold * linear_reg) |
        NotComponent.left("AutoARIMA")
    )
    condition.pipeline = complex_pipeline
    assert condition.should_enter(
        time=current_time,
        strategy=strategy,
        manager=manager
    ), "Should enter even if ARIMA model fails due to OR condition"

def test_parameter_validation():
    with pytest.raises(ValueError):
        # Invalid RSI period type
        RSIComponent(period="14 days", oversold=30)
        
    with pytest.raises(ValueError):
        # Invalid component type
        CompositePipelineCondition(
            pipeline="not a component",
            ohlc_data="dummy_data"
        )

if __name__ == "__main__":
    pytest.main(["-v", __file__])
