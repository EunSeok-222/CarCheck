"""
견적서 DB → ChromaDB 벡터 인덱스 빌드 (최초 1회 실행)
python3 utils/build_rag.py
"""
import sqlite3
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_PATH  = Path(__file__).parent.parent / "data" / "estimates.db"
RAG_PATH = Path(__file__).parent.parent / "data" / "rag_db"
SAMPLE   = 8000  # 임베딩할 사례 수

def build():
    print("📦 견적서 DB에서 수리 사례 로드 중...")
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("""
        SELECT
            e.manufacturer, e.car_name,
            r.part_name, r.action,
            CAST(r.labor_cost AS INTEGER) as labor_cost,
            CAST(r.labor_cost / (1 - NULLIF(r.hq_percent,0)/100.0) AS INTEGER) as total_cost
        FROM repair_items r
        JOIN estimates e ON r.estimate_id = e.estimate_id
        WHERE r.part_name IS NOT NULL
          AND r.action   IS NOT NULL
          AND r.labor_cost > 0
        ORDER BY RANDOM()
        LIMIT ?
    """, (SAMPLE,)).fetchall()
    con.close()
    print(f"✅ {len(rows)}건 로드 완료")

    print("🔤 텍스트 문서 변환 중...")
    documents, metadatas, ids = [], [], []
    for i, (mfr, car, part, action, labor, total) in enumerate(rows):
        text = f"부위: {part} | 작업: {action} | 차종: {mfr or ''} {car or ''} | 공임: {labor:,}원 | 예상비용: {total:,}원"
        documents.append(text)
        metadatas.append({
            "part":   part,
            "action": action,
            "car":    f"{mfr or ''} {car or ''}".strip(),
            "labor":  labor,
            "total":  total,
        })
        ids.append(f"case_{i}")

    print("🧠 임베딩 + ChromaDB 저장 중 (수 분 소요)...")
    ef = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path=str(RAG_PATH))

    try:
        client.delete_collection("repair_cases")
    except Exception:
        pass
    col = client.create_collection("repair_cases", embedding_function=ef)

    batch = 500
    for start in range(0, len(documents), batch):
        col.add(
            documents=documents[start:start+batch],
            metadatas=metadatas[start:start+batch],
            ids=ids[start:start+batch],
        )
        print(f"  {min(start+batch, len(documents))}/{len(documents)} 저장...")

    print(f"\n🎉 RAG 인덱스 빌드 완료 → {RAG_PATH}")
    print(f"   총 {col.count()}건 저장됨")

if __name__ == "__main__":
    build()
