import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "estimates.db"
DATA_REPO_ID = "eunseok22/carcheck-data"


def _ensure_db():
    """estimates.db(112MB, GitHub 100MB 제한 초과)가 없으면 HF Hub 데이터셋에서 내려받는다."""
    if DB_PATH.exists():
        return
    try:
        from huggingface_hub import hf_hub_download
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=DATA_REPO_ID, filename="estimates.db",
                        repo_type="dataset", local_dir=str(DB_PATH.parent))
    except Exception:
        pass

# YOLO 손상 유형 → 작업 키워드 매핑
_DAMAGE_TO_ACTION = {
    "Scratched":  "도장",
    "Breakage":   "교환",
    "Crushed":    "도장",   # 판금 후 도장
    "Separated":  "탈착",
}

# YOLO 부품명 → DB 검색 키워드 매핑
_PART_KEYWORDS = {
    "앞 범퍼":       ["범퍼", "후론트 범퍼"],
    "뒤 범퍼":       ["리어 범퍼"],
    "헤드라이트(좌)": ["헤드램프", "전조등"],
    "헤드라이트(우)": ["헤드램프", "전조등"],
    "앞 도어(좌)":   ["앞문", "앞 도어", "프론트 도어"],
    "앞 도어(우)":   ["앞문", "앞 도어", "프론트 도어"],
    "뒤 도어(좌)":   ["뒷문", "리어 도어"],
    "뒤 도어(우)":   ["뒷문", "리어 도어"],
    "앞 펜더(좌)":   ["앞 휀다", "프론트 펜더"],
    "앞 펜더(우)":   ["앞 휀다", "프론트 펜더"],
    "트렁크":        ["트렁크"],
    "본네트":        ["후드", "본네트"],
}

# 사진 업로드 시 사용자가 직접 고르는 촬영 부위 선택지.
# YOLO 모델은 손상 유형(긁힘/파손/분리/찌그러짐)만 분류하고 부위는 예측하지
# 않으므로, 부위는 사용자 입력으로 받아 위 _PART_KEYWORDS와 동일한 이름 체계를 쓴다.
PART_OPTIONS = list(_PART_KEYWORDS.keys())

_DEFAULT_COST = 120_000  # DB에서 못 찾을 때 기본값


def _get_cost_from_db(part: str, action_ko: str) -> int:
    """part_cost_avg 뷰에서 평균 공임 조회. 못 찾으면 _DEFAULT_COST."""
    _ensure_db()
    keywords = _PART_KEYWORDS.get(part, [part])

    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()

        for kw in keywords:
            row = cur.execute(
                """
                SELECT avg_cost FROM part_cost_avg
                WHERE part_name LIKE ? AND action LIKE ?
                ORDER BY sample_count DESC
                LIMIT 1
                """,
                (f"%{kw}%", f"%{action_ko}%"),
            ).fetchone()
            if row:
                con.close()
                return row[0]

        # 작업 무관하게 부품만으로 재검색
        for kw in keywords:
            row = cur.execute(
                """
                SELECT avg_cost FROM part_cost_avg
                WHERE part_name LIKE ?
                ORDER BY sample_count DESC
                LIMIT 1
                """,
                (f"%{kw}%",),
            ).fetchone()
            if row:
                con.close()
                return row[0]

        con.close()
    except Exception:
        pass

    return _DEFAULT_COST


def estimate_repair_cost(damage_result: dict) -> dict:
    breakdown = []

    for dmg in damage_result["damages"]:
        part      = dmg["part"]
        action_en = dmg["type"]
        action_ko = _DAMAGE_TO_ACTION.get(action_en, "교환")
        cost      = _get_cost_from_db(part, action_ko)

        breakdown.append({
            "part":   part,
            "action": action_en,
            "cost":   cost,
        })

    total = sum(item["cost"] for item in breakdown)
    return {"total": total, "breakdown": breakdown}
