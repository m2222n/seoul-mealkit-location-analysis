"""
네이버 지도 밀키트 매장 리뷰 크롤러

Selenium을 활용하여 네이버 지도에서 밀키트 매장의 리뷰 데이터를 수집합니다.
- 매장별 리뷰어 ID, 방문횟수 추출
- 1인당 평균 방문수 계산 (타겟 변수 생성용)
"""

import time
import re
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException
)


@dataclass
class ReviewData:
    """리뷰 데이터 구조"""
    store_name: str
    reviewer_id: str
    visit_count: int
    rating: Optional[float] = None
    review_text: Optional[str] = None


class NaverMapCrawler:
    """네이버 지도 밀키트 매장 크롤러"""

    NAVER_MAP_URL = "https://map.naver.com/v5/search"

    def __init__(self, headless: bool = True):
        """
        Args:
            headless: 브라우저 숨김 모드 여부
        """
        self.driver = self._setup_driver(headless)
        self.wait = WebDriverWait(self.driver, 10)

    def _setup_driver(self, headless: bool) -> webdriver.Chrome:
        """Chrome 드라이버 설정"""
        options = Options()

        if headless:
            options.add_argument('--headless')

        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-gpu')
        # 일반적인 User-Agent 사용
        options.add_argument(
            '--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )

        return webdriver.Chrome(options=options)

    def search_stores(self, keyword: str = "서울 밀키트") -> List[str]:
        """
        키워드로 매장 검색

        Args:
            keyword: 검색 키워드

        Returns:
            매장 이름 리스트
        """
        self.driver.get(self.NAVER_MAP_URL)
        time.sleep(2)

        # 검색창에 키워드 입력
        search_input = self.wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "input.input_search"))
        )
        search_input.clear()
        search_input.send_keys(keyword)
        search_input.send_keys(Keys.ENTER)
        time.sleep(3)

        # 검색 결과 iframe으로 전환
        self._switch_to_search_iframe()

        store_names = []

        # 스크롤하면서 모든 매장 수집
        while True:
            stores = self.driver.find_elements(By.CSS_SELECTOR, "span.place_bluelink")

            for store in stores:
                name = store.text.strip()
                if name and name not in store_names:
                    store_names.append(name)

            # 스크롤 다운
            if not self._scroll_down():
                break

        self.driver.switch_to.default_content()
        return store_names

    def _switch_to_search_iframe(self) -> None:
        """검색 결과 iframe으로 전환"""
        iframe = self.wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "iframe#searchIframe"))
        )
        self.driver.switch_to.frame(iframe)

    def _switch_to_entry_iframe(self) -> None:
        """매장 상세 iframe으로 전환"""
        self.driver.switch_to.default_content()
        iframe = self.wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "iframe#entryIframe"))
        )
        self.driver.switch_to.frame(iframe)

    def _scroll_down(self) -> bool:
        """스크롤 다운, 더 스크롤 가능하면 True 반환"""
        try:
            scrollable = self.driver.find_element(By.CSS_SELECTOR, "div.Ryr1F")
            last_height = self.driver.execute_script(
                "return arguments[0].scrollHeight", scrollable
            )

            self.driver.execute_script(
                "arguments[0].scrollTop = arguments[0].scrollHeight", scrollable
            )
            time.sleep(1)

            new_height = self.driver.execute_script(
                "return arguments[0].scrollHeight", scrollable
            )

            return new_height > last_height
        except:
            return False

    def get_store_reviews(self, store_name: str) -> List[ReviewData]:
        """
        특정 매장의 리뷰 데이터 수집

        Args:
            store_name: 매장 이름

        Returns:
            리뷰 데이터 리스트
        """
        reviews = []

        try:
            # 매장 검색 및 클릭
            self._click_store(store_name)
            time.sleep(2)

            # 매장 상세 iframe으로 전환
            self._switch_to_entry_iframe()

            # 리뷰 탭 클릭
            self._click_review_tab()
            time.sleep(2)

            # 리뷰 수집
            reviews = self._extract_reviews(store_name)

        except Exception as e:
            print(f"[ERROR] {store_name} 리뷰 수집 실패: {e}")

        finally:
            self.driver.switch_to.default_content()

        return reviews

    def _click_store(self, store_name: str) -> None:
        """매장 클릭"""
        self._switch_to_search_iframe()

        stores = self.driver.find_elements(By.CSS_SELECTOR, "span.place_bluelink")
        for store in stores:
            if store_name in store.text:
                store.click()
                break

        self.driver.switch_to.default_content()

    def _click_review_tab(self) -> None:
        """리뷰 탭 클릭"""
        tabs = self.driver.find_elements(By.CSS_SELECTOR, "a.veBoZ")
        for tab in tabs:
            if "리뷰" in tab.text:
                tab.click()
                break

    def _extract_reviews(self, store_name: str) -> List[ReviewData]:
        """리뷰 데이터 추출"""
        reviews = []

        # 모든 리뷰 로드 (더보기 클릭)
        self._load_all_reviews()

        review_elements = self.driver.find_elements(
            By.CSS_SELECTOR, "li.pui__X35jYm"
        )

        for elem in review_elements:
            try:
                # 리뷰어 ID 추출
                reviewer_elem = elem.find_element(By.CSS_SELECTOR, "span.pui__NMi-Dp")
                reviewer_id = reviewer_elem.text.strip()

                # 방문횟수 추출 (예: "5번째 방문")
                visit_count = 1
                try:
                    visit_elem = elem.find_element(By.CSS_SELECTOR, "span.pui__QKE5Pr")
                    visit_text = visit_elem.text
                    match = re.search(r'(\d+)번째', visit_text)
                    if match:
                        visit_count = int(match.group(1))
                except NoSuchElementException:
                    pass

                # 별점 추출 (선택적)
                rating = None
                try:
                    rating_elem = elem.find_element(By.CSS_SELECTOR, "span.pui__XuaVHi")
                    rating = float(rating_elem.text)
                except (NoSuchElementException, ValueError):
                    pass

                reviews.append(ReviewData(
                    store_name=store_name,
                    reviewer_id=reviewer_id,
                    visit_count=visit_count,
                    rating=rating
                ))

            except StaleElementReferenceException:
                continue
            except Exception as e:
                continue

        return reviews

    def _load_all_reviews(self, max_clicks: int = 50) -> None:
        """더보기 버튼 클릭하여 모든 리뷰 로드"""
        for _ in range(max_clicks):
            try:
                more_btn = self.driver.find_element(
                    By.CSS_SELECTOR, "a.pui__tzwrk0"
                )
                if more_btn.is_displayed():
                    more_btn.click()
                    time.sleep(0.5)
                else:
                    break
            except NoSuchElementException:
                break
            except:
                break

    def crawl_all_stores(self, keyword: str = "서울 밀키트") -> pd.DataFrame:
        """
        모든 매장의 리뷰 데이터 수집

        Args:
            keyword: 검색 키워드

        Returns:
            전체 리뷰 데이터프레임
        """
        print(f"[INFO] '{keyword}' 검색 중...")
        store_names = self.search_stores(keyword)
        print(f"[INFO] {len(store_names)}개 매장 발견")

        all_reviews = []

        for i, store_name in enumerate(store_names, 1):
            print(f"[INFO] ({i}/{len(store_names)}) {store_name} 크롤링 중...")
            reviews = self.get_store_reviews(store_name)
            all_reviews.extend(reviews)
            print(f"  -> {len(reviews)}개 리뷰 수집")
            time.sleep(1)  # 요청 간격 조절

        # DataFrame 변환
        df = pd.DataFrame([
            {
                'store_name': r.store_name,
                'reviewer_id': r.reviewer_id,
                'visit_count': r.visit_count,
                'rating': r.rating
            }
            for r in all_reviews
        ])

        return df

    def close(self) -> None:
        """브라우저 종료"""
        if self.driver:
            self.driver.quit()


def calculate_avg_visits(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    매장별 1인당 평균 방문수 계산

    Target = 총 방문횟수 / 리뷰어 ID 수

    Args:
        reviews_df: 리뷰 데이터프레임

    Returns:
        매장별 평균 방문수 데이터프레임
    """
    if reviews_df.empty:
        return pd.DataFrame(columns=['store_name', 'total_visits', 'unique_visitors', 'avg_visits'])

    store_stats = reviews_df.groupby('store_name').agg(
        total_visits=('visit_count', 'sum'),
        unique_visitors=('reviewer_id', 'nunique')
    ).reset_index()

    store_stats['avg_visits'] = store_stats['total_visits'] / store_stats['unique_visitors']

    return store_stats.sort_values('avg_visits', ascending=False)


# 사용 예시
if __name__ == "__main__":
    from pathlib import Path

    # 출력 경로 설정
    OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    crawler = NaverMapCrawler(headless=False)

    try:
        # 전체 크롤링
        reviews_df = crawler.crawl_all_stores("서울 밀키트")

        # 저장
        reviews_df.to_csv(
            OUTPUT_DIR / "mealkit_reviews.csv",
            index=False,
            encoding='utf-8-sig'
        )

        # 1인당 평균 방문수 계산
        target_df = calculate_avg_visits(reviews_df)
        target_df.to_csv(
            OUTPUT_DIR / "mealkit_target.csv",
            index=False,
            encoding='utf-8-sig'
        )

        print("\n[결과]")
        print(target_df.head(10))

    finally:
        crawler.close()
