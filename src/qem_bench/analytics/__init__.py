"""
Advanced Analytics Module for QEM-Bench

Real-time analytics, data visualization, and intelligent insights
for quantum error mitigation research and production systems.
"""

# Core analytics components
from .real_time_analytics import (
    RealTimeAnalyzer, AnalyticsEngine, MetricsProcessor,
    TrendAnalyzer, AnomalyDetector, PatternRecognizer
)

# Data visualization and dashboards
from .visualization import (
    InteractiveDashboard, PlotManager, DataVisualizer,
    QuantumMetricsPlotter, ComparisonAnalyzer, ReportGenerator
)

# Machine learning insights
from .ml_insights import (
    InsightEngine, PredictiveAnalyzer, ClusterAnalyzer,
    PerformancePredictor, OptimizationRecommender, TrendForecaster
)

# Research analytics
from .research_analytics import (
    ResearchAnalyzer, ExperimentTracker, ResultsAggregator,
    PublicationAnalytics, CollaborationTracker, ImpactAnalyzer
)

__all__ = [
    # Real-time analytics
    "RealTimeAnalyzer", "AnalyticsEngine", "MetricsProcessor",
    "TrendAnalyzer", "AnomalyDetector", "PatternRecognizer",
    
    # Visualization
    "InteractiveDashboard", "PlotManager", "DataVisualizer",
    "QuantumMetricsPlotter", "ComparisonAnalyzer", "ReportGenerator",
    
    # ML insights
    "InsightEngine", "PredictiveAnalyzer", "ClusterAnalyzer",
    "PerformancePredictor", "OptimizationRecommender", "TrendForecaster",
    
    # Research analytics
    "ResearchAnalyzer", "ExperimentTracker", "ResultsAggregator",
    "PublicationAnalytics", "CollaborationTracker", "ImpactAnalyzer"
]