"""
서울시 밀키트 매장 입지 분석 패키지
"""

from .data_loader import SeoulOpenDataLoader
from .preprocessing import DataPreprocessor
from .features import FeatureEngineer
from .models import MealkitLocationModel
from .visualization import ResultVisualizer

# Crawler는 selenium 설치 시에만 사용 가능
try:
    from .crawler import NaverMapCrawler, calculate_avg_visits
    _CRAWLER_AVAILABLE = True
except ImportError:
    NaverMapCrawler = None
    calculate_avg_visits = None
    _CRAWLER_AVAILABLE = False

__version__ = "1.0.0"
__all__ = [
    "SeoulOpenDataLoader",
    "DataPreprocessor",
    "FeatureEngineer",
    "MealkitLocationModel",
    "ResultVisualizer",
    "NaverMapCrawler",
    "calculate_avg_visits"
]
