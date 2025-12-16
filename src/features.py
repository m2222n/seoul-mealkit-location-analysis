"""
피처 엔지니어링 모듈
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from sklearn.decomposition import PCA


# 기본 키워드 설정 (한글/영문 모두 지원)
DEFAULT_KEYWORDS = {
    'age_30_59': ['30대', '40대', '50대', '30~', '40~', '50~', '30s', '40s', '50s'],
    'small_household': ['1인', '2인', '1~2인', 'single', 'one_person', 'two_person'],
    'dual_income': ['맞벌이', 'dual_income', 'double_income'],
    'transport': ['지하철', '버스', 'subway', 'bus', 'metro', 'transport']
}


class FeatureEngineer:
    """밀키트 입지 분석용 피처 엔지니어링"""

    def __init__(self, keywords: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            keywords: 피처 생성에 사용할 키워드 딕셔너리
                      None이면 DEFAULT_KEYWORDS 사용
        """
        self.pca = None
        self.pca_columns = None
        self.keywords = keywords or DEFAULT_KEYWORDS

    def _find_columns(self, df: pd.DataFrame, keyword_key: str) -> List[str]:
        """키워드에 해당하는 컬럼 찾기"""
        keywords = self.keywords.get(keyword_key, [])
        return [
            col for col in df.columns
            if any(kw in col for kw in keywords)
        ]

    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """연령대별 피처 생성"""
        df = df.copy()

        age_cols = self._find_columns(df, 'age_30_59')
        if age_cols:
            df['target_age_pop'] = df[age_cols].sum(axis=1)

        return df

    def create_household_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """가구 유형 피처 생성"""
        df = df.copy()

        # 소형 가구 (1-2인)
        small_cols = self._find_columns(df, 'small_household')
        if small_cols:
            df['small_household'] = df[small_cols].sum(axis=1)

        # 맞벌이 가구
        dual_cols = self._find_columns(df, 'dual_income')
        if dual_cols:
            df['dual_income'] = df[dual_cols].sum(axis=1)

        return df

    def create_infrastructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """인프라 관련 피처 생성"""
        df = df.copy()

        transport_cols = self._find_columns(df, 'transport')
        if transport_cols:
            df['transport_access'] = df[transport_cols].sum(axis=1)

        return df

    def apply_pca(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_components: int = 5,
        prefix: str = 'pca'
    ) -> pd.DataFrame:
        """PCA 차원 축소 적용"""
        df = df.copy()

        # 컬럼 존재 여부 확인
        missing_cols = [c for c in columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"존재하지 않는 컬럼: {missing_cols}")

        self.pca = PCA(n_components=n_components)
        self.pca_columns = columns

        pca_result = self.pca.fit_transform(df[columns])

        # PCA 결과를 데이터프레임에 추가
        pca_cols = [f'{prefix}_{i+1}' for i in range(n_components)]
        pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=df.index)

        # 원본 컬럼 제거 후 PCA 결과 추가
        df = df.drop(columns=columns)
        df = pd.concat([df, pca_df], axis=1)

        return df

    def transform_pca(self, df: pd.DataFrame, prefix: str = 'pca') -> pd.DataFrame:
        """학습된 PCA로 새 데이터 변환"""
        if self.pca is None or self.pca_columns is None:
            raise ValueError("먼저 apply_pca()를 실행하세요.")

        df = df.copy()
        pca_result = self.pca.transform(df[self.pca_columns])

        pca_cols = [f'{prefix}_{i+1}' for i in range(self.pca.n_components_)]
        pca_df = pd.DataFrame(pca_result, columns=pca_cols, index=df.index)

        df = df.drop(columns=self.pca_columns)
        df = pd.concat([df, pca_df], axis=1)

        return df

    def get_pca_variance_ratio(self) -> Optional[np.ndarray]:
        """PCA 설명 분산 비율 반환"""
        if self.pca is None:
            return None
        return self.pca.explained_variance_ratio_

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 피처 엔지니어링 적용"""
        df = self.create_age_features(df)
        df = self.create_household_features(df)
        df = self.create_infrastructure_features(df)
        return df


def get_feature_importance(
    model,
    feature_names: List[str]
) -> pd.DataFrame:
    """모델의 피처 중요도 추출"""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        raise ValueError("모델에서 피처 중요도를 추출할 수 없습니다.")

    return pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
