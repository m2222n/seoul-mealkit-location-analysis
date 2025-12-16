"""
서울시 공공데이터 API 로더
"""

import requests
import pandas as pd
from typing import Optional, Dict, List


class SeoulOpenDataLoader:
    """서울시 열린데이터광장 API를 통한 데이터 수집"""

    BASE_URL = "http://openapi.seoul.go.kr:8088"

    def __init__(self, api_key: str):
        """
        Args:
            api_key: 서울시 공공데이터 API 키
        """
        self.api_key = api_key

    def _make_request(self, service: str, start: int = 1, end: int = 1000) -> Dict:
        """API 요청 실행"""
        url = f"{self.BASE_URL}/{self.api_key}/json/{service}/{start}/{end}/"

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            raise ConnectionError(f"API 요청 시간 초과: {service}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API 요청 실패: {e}")

    def get_subway_passengers(self) -> pd.DataFrame:
        """행정동별 지하철 승객 수 조회"""
        data = self._make_request("tpssSubwayPassenger")

        records = []
        for row in data['tpssSubwayPassenger']['row']:
            records.append({
                'district_id': row['ADMDONG_ID'],
                'subway_passengers': row['SBWY_PSGR_CNT']
            })

        df = pd.DataFrame(records)
        return df.groupby('district_id')['subway_passengers'].sum().reset_index()

    def get_bus_passengers(self) -> pd.DataFrame:
        """행정동별 버스 승객 수 조회"""
        data = self._make_request("tpssEmdBus")

        records = []
        for row in data['tpssEmdBus']['row']:
            records.append({
                'district_id': row['ADMDONG_ID'],
                'bus_passengers': row['BUS_PSGR_CNT']
            })

        df = pd.DataFrame(records)
        return df.groupby('district_id')['bus_passengers'].sum().reset_index()

    def get_floating_population(self) -> pd.DataFrame:
        """행정동별 유동인구 조회"""
        data = self._make_request("tpssEmdFloatPop")

        records = []
        for row in data.get('tpssEmdFloatPop', {}).get('row', []):
            records.append({
                'district_id': row.get('ADMDONG_ID'),
                'floating_pop': row.get('FLPOP_CNT')
            })

        return pd.DataFrame(records)

    def load_all_transport_data(self) -> pd.DataFrame:
        """교통 관련 데이터 통합 로드"""
        subway = self.get_subway_passengers()
        bus = self.get_bus_passengers()

        merged = pd.merge(subway, bus, on='district_id', how='outer')
        merged.fillna(0, inplace=True)

        return merged


def load_csv(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """CSV 파일 로드 (여러 인코딩 시도)"""
    encodings = [encoding, 'cp949', 'euc-kr', 'utf-8-sig']

    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"파일을 읽을 수 없습니다: {filepath}")


def load_excel(filepath: str, sheet_name: int = 0) -> pd.DataFrame:
    """Excel 파일 로드"""
    return pd.read_excel(filepath, sheet_name=sheet_name)
