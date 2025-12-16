"""
머신러닝/딥러닝 모델 모듈
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


class MealkitLocationModel:
    """밀키트 매장 입지 예측 모델"""

    def __init__(self, task: str = 'classification'):
        """
        Args:
            task: 'classification' 또는 'regression'
        """
        self.task = task
        self.model = None
        self.best_params = None
        self.feature_names = None

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        **kwargs
    ) -> 'MealkitLocationModel':
        """Random Forest 모델 학습"""
        self.feature_names = list(X_train.columns)

        if self.task == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                **kwargs
            )

        self.model.fit(X_train, y_train)
        return self

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        **kwargs
    ) -> 'MealkitLocationModel':
        """LightGBM 모델 학습"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm 패키지를 설치하세요: pip install lightgbm")

        self.feature_names = list(X_train.columns)

        if self.task == 'classification':
            self.model = lgb.LGBMClassifier(**kwargs)
        else:
            self.model = lgb.LGBMRegressor(**kwargs)

        self.model.fit(X_train, y_train)
        return self

    def train_deep_learning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        epochs: int = 100,
        batch_size: int = 8,
        validation_split: float = 0.2,
        verbose: int = 0
    ) -> 'MealkitLocationModel':
        """딥러닝 모델 학습"""
        try:
            from tensorflow.keras.layers import Input, Dense, Dropout
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.callbacks import EarlyStopping
        except ImportError:
            raise ImportError("tensorflow 패키지를 설치하세요: pip install tensorflow")

        self.feature_names = list(X_train.columns)
        input_dim = X_train.shape[1]

        # 모델 구조
        self.model = Sequential([
            Input(shape=(input_dim,)),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1, activation='linear' if self.task == 'regression' else 'sigmoid')
        ])

        # 컴파일
        loss = 'mse' if self.task == 'regression' else 'binary_crossentropy'
        metrics = ['mae'] if self.task == 'regression' else ['accuracy']

        self.model.compile(optimizer='adam', loss=loss, metrics=metrics)

        # Early Stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # 학습
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        if self.model is None:
            raise ValueError("먼저 모델을 학습하세요.")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """확률 예측 (분류 모델)"""
        if self.task != 'classification':
            raise ValueError("분류 모델에서만 사용 가능합니다.")
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """모델 평가"""
        y_pred = self.predict(X_test)

        if self.task == 'classification':
            # 딥러닝 모델의 경우 확률 → 클래스 변환
            if hasattr(y_pred, 'flatten') and y_pred.ndim > 1:
                y_pred = (y_pred.flatten() > 0.5).astype(int)

            return {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            y_pred = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
            return {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """피처 중요도 반환"""
        if self.model is None or self.feature_names is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

        return None

    def save(self, filepath: str) -> None:
        """모델 저장"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'task': self.task,
                'feature_names': self.feature_names
            }, f)

    def load(self, filepath: str) -> 'MealkitLocationModel':
        """모델 로드"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.task = data['task']
            self.feature_names = data['feature_names']
        return self


def cross_validate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = 'accuracy'
) -> Dict[str, float]:
    """교차 검증"""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores.tolist()
    }
