"""Turing Tumble Board Analytics Package.

Provides feature extraction, correlation analysis, visualization, and
reporting for the Turing Tumble benchmark.
"""

from .board_analytics import (  # noqa: F401, E402
    BoardFeatureExtractor,
    BenchmarkIntegrator,
    CorrelationAnalyzer,
    BoardVisualizer,
    ReportGenerator,
    run_pipeline,
    run_aggregated_pipeline,
)

__all__ = [
    "BoardFeatureExtractor",
    "BenchmarkIntegrator",
    "CorrelationAnalyzer",
    "BoardVisualizer",
    "ReportGenerator",
    "run_pipeline",
    "run_aggregated_pipeline",
]
