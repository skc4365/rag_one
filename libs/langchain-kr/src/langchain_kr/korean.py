"""한국어 불용어(stopwords) 유틸리티.

원격 GitHub 저장소를 매번 내려받던 기존 구현을 대체합니다.
불용어 목록은 패키지에 함께 배포되는 ``assets/korean_stopwords.txt`` 에서 읽습니다.
"""

from functools import lru_cache
from importlib import resources
from typing import List


@lru_cache(maxsize=1)
def _load() -> tuple:
    text = (
        resources.files(__package__)
        .joinpath("assets/korean_stopwords.txt")
        .read_text(encoding="utf-8")
    )
    return tuple(word.strip() for word in text.splitlines() if word.strip())


def stopwords() -> List[str]:
    """패키지에 내장된 한국어 불용어 목록을 반환합니다."""
    return list(_load())
