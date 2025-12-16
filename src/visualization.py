"""
시각화 모듈
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from pathlib import Path


# 한글 폰트 설정 (OS별 자동 감지)
import platform

def set_korean_font():
    """OS에 맞는 한글 폰트 설정"""
    system = platform.system()

    if system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'AppleGothic'
    elif system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'NanumGothic'

    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()


class ResultVisualizer:
    """분석 결과 시각화"""

    def __init__(self, figsize: Tuple[int, int] = (10, 6)):
        self.figsize = figsize
        self.output_dir = None

    def set_output_dir(self, path: str) -> None:
        """출력 디렉토리 설정"""
        self.output_dir = Path(path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
        title: str = "Feature Importance",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """피처 중요도 시각화"""
        fig, ax = plt.subplots(figsize=self.figsize)

        data = importance_df.head(top_n).sort_values('importance')

        ax.barh(data['feature'], data['importance'], color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_district_comparison(
        self,
        top_df: pd.DataFrame,
        bottom_df: pd.DataFrame,
        columns: List[str],
        title: str = "상위 vs 하위 지역 비교",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """상위/하위 지역 피처 비교"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # 평균값 계산
        top_mean = top_df[columns].mean()
        bottom_mean = bottom_df[columns].mean()

        # 데이터 준비
        x = np.arange(len(columns))
        width = 0.35

        ax.bar(x - width/2, top_mean, width, label='상위 지역', color='#2ecc71')
        ax.bar(x + width/2, bottom_mean, width, label='하위 지역', color='#e74c3c')

        ax.set_ylabel('평균값 (정규화)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(columns, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_prediction_ranking(
        self,
        df: pd.DataFrame,
        district_col: str,
        score_col: str,
        top_n: int = 15,
        title: str = "예측 점수 상위 지역",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """예측 점수 순위 시각화"""
        fig, ax = plt.subplots(figsize=self.figsize)

        data = df.nlargest(top_n, score_col).sort_values(score_col)

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(data)))
        ax.barh(data[district_col], data[score_col], color=colors)

        ax.set_xlabel('예측 점수')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_correlation_heatmap(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "상관관계 히트맵",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """상관관계 히트맵"""
        fig, ax = plt.subplots(figsize=(12, 10))

        if columns:
            corr = df[columns].corr()
        else:
            corr = df.select_dtypes(include=[np.number]).corr()

        mask = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            ax=ax
        )
        ax.set_title(title)

        plt.tight_layout()

        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_model_comparison(
        self,
        results: dict,
        metric: str = 'accuracy',
        title: str = "모델 성능 비교",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """모델별 성능 비교"""
        fig, ax = plt.subplots(figsize=self.figsize)

        models = list(results.keys())
        scores = [results[m].get(metric, 0) for m in models]

        colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(models)))
        bars = ax.bar(models, scores, color=colors)

        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

        # 값 표시
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{score:.3f}',
                ha='center',
                va='bottom'
            )

        plt.tight_layout()

        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')

        return fig

    def plot_pca_variance(
        self,
        variance_ratio: np.ndarray,
        title: str = "PCA 설명 분산 비율",
        save_name: Optional[str] = None
    ) -> plt.Figure:
        """PCA 설명 분산 시각화"""
        fig, ax = plt.subplots(figsize=self.figsize)

        n_components = len(variance_ratio)
        x = range(1, n_components + 1)
        cumulative = np.cumsum(variance_ratio)

        ax.bar(x, variance_ratio, alpha=0.7, label='개별 분산')
        ax.plot(x, cumulative, 'ro-', label='누적 분산')

        ax.set_xlabel('주성분')
        ax.set_ylabel('설명 분산 비율')
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_name and self.output_dir:
            fig.savefig(self.output_dir / save_name, dpi=150, bbox_inches='tight')

        return fig
