"""
데이터 전처리 모듈
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataPreprocessor:
    """밀키트 입지 분석용 데이터 전처리"""

    def __init__(self):
        self.scaler = None
        self.feature_columns = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 데이터 정제"""
        df = df.copy()

        # 중복 제거
        df = df.drop_duplicates()

        # 컬럼명 정리 (공백 제거)
        df.columns = df.columns.str.strip()

        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'mean',
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        결측치 처리

        Args:
            df: 입력 데이터프레임
            strategy: 'mean', 'median', 'zero', 'drop', 'fill'
            fill_value: strategy='fill'일 때 사용할 값
        """
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif strategy == 'drop':
            df = df.dropna()
        elif strategy == 'fill' and fill_value is not None:
            df[numeric_cols] = df[numeric_cols].fillna(fill_value)

        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        피처 스케일링

        Args:
            df: 입력 데이터프레임
            columns: 스케일링할 컬럼 목록
            method: 'standard' (Z-score) 또는 'minmax'
        """
        df = df.copy()

        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        df[columns] = self.scaler.fit_transform(df[columns])
        self.feature_columns = columns

        return df

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습된 스케일러로 새 데이터 변환"""
        if self.scaler is None or self.feature_columns is None:
            raise ValueError("먼저 scale_features()를 실행하세요.")

        df = df.copy()
        df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """범주형 변수 인코딩"""
        df = df.copy()

        if method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
        elif method == 'label':
            for col in columns:
                df[col] = df[col].astype('category').cat.codes

        return df


def merge_datasets(
    base_df: pd.DataFrame,
    *dfs: pd.DataFrame,
    on: str,
    how: str = 'left'
) -> pd.DataFrame:
    """여러 데이터프레임 병합"""
    result = base_df.copy()

    for df in dfs:
        result = pd.merge(result, df, on=on, how=how)

    return result


def split_train_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """학습/테스트 데이터 분리"""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
