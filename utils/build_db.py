"""
견적서 JSON 125,006개 → SQLite DB 변환 스크립트
실행: python3 utils/build_db.py
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

JSON_DIR = Path(
    "/private/tmp/claude-501/-Users-eunseoklee"
    "/b707da8f-fb86-498b-8c35-3e2d893f30c8/scratchpad/estimates_unzip"
)
DB_PATH = Path(__file__).parent.parent / "data" / "estimates.db"


def parse_cost(value: str) -> int:
    """'40,920' → 40920, 빈 문자열 → 0"""
    if not value:
        return 0
    cleaned = re.sub(r"[^0-9]", "", str(value))
    return int(cleaned) if cleaned else 0


def build_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE estimates (
            estimate_id  TEXT PRIMARY KEY,
            manufacturer TEXT,
            car_name     TEXT,
            model        TEXT,
            total_cost   INTEGER,
            labor_cost   INTEGER,
            parts_cost   INTEGER
        );

        CREATE TABLE repair_items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            estimate_id TEXT REFERENCES estimates(estimate_id),
            part_name   TEXT,
            action      TEXT,
            labor_cost  INTEGER,
            hq_percent  REAL
        );

        CREATE INDEX idx_part_action ON repair_items (part_name, action);
    """)

    files = sorted(JSON_DIR.glob("*.json"))
    total = len(files)
    print(f"파일 수: {total:,}개")

    ok = skip = 0
    batch_estimates = []
    batch_items = []
    BATCH = 2000

    for i, path in enumerate(files, 1):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skip += 1
            continue

        est_id = path.stem  # as-XXXXXXX

        info  = data.get("차량정보", {})
        money = data.get("수리비 정산정보", {})
        total_cost  = parse_cost(money.get("합계", {}).get("총계", ""))
        labor_cost  = parse_cost(money.get("공임", {}).get("공임소계", ""))
        parts_cost  = parse_cost(money.get("부품", {}).get("부품소계", ""))

        batch_estimates.append((
            est_id,
            info.get("제작사/차종", ""),
            info.get("차량명칭", ""),
            info.get("모델", ""),
            total_cost,
            labor_cost,
            parts_cost,
        ))

        for item in data.get("수리내역", []):
            lc = parse_cost(item.get("공임", ""))
            if not lc:
                continue
            batch_items.append((
                est_id,
                item.get("작업항목 및 부품명", "").strip(),
                item.get("작업", "").strip(),
                lc,
                item.get("HQ%", 0.0) or 0.0,
            ))

        ok += 1

        if i % BATCH == 0:
            cur.executemany(
                "INSERT OR IGNORE INTO estimates VALUES (?,?,?,?,?,?,?)",
                batch_estimates,
            )
            cur.executemany(
                "INSERT INTO repair_items (estimate_id,part_name,action,labor_cost,hq_percent)"
                " VALUES (?,?,?,?,?)",
                batch_items,
            )
            con.commit()
            batch_estimates.clear()
            batch_items.clear()
            pct = i / total * 100
            print(f"  {i:>7,}/{total:,}  ({pct:.1f}%)", end="\r", flush=True)

    # 나머지
    if batch_estimates:
        cur.executemany(
            "INSERT OR IGNORE INTO estimates VALUES (?,?,?,?,?,?,?)",
            batch_estimates,
        )
        cur.executemany(
            "INSERT INTO repair_items (estimate_id,part_name,action,labor_cost,hq_percent)"
            " VALUES (?,?,?,?,?)",
            batch_items,
        )
        con.commit()

    # 부품별 평균 단가 뷰
    cur.executescript("""
        CREATE VIEW part_cost_avg AS
        SELECT
            part_name,
            action,
            COUNT(*)           AS sample_count,
            CAST(AVG(labor_cost) AS INTEGER) AS avg_cost,
            CAST(MIN(labor_cost) AS INTEGER) AS min_cost,
            CAST(MAX(labor_cost) AS INTEGER) AS max_cost
        FROM repair_items
        WHERE labor_cost > 0
        GROUP BY part_name, action
        HAVING COUNT(*) >= 3;
    """)
    con.commit()
    con.close()

    print(f"\n완료: 성공 {ok:,}개 / 스킵 {skip:,}개")
    print(f"DB: {DB_PATH}  ({DB_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    build_db()
