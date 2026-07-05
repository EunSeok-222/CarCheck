"""
보험 약관·할증·법률 PDF → ChromaDB 지식 인덱스 빌드 (문서 기반 RAG)
python3 utils/build_knowledge_rag.py
"""
import re
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

KNOWLEDGE_DIR = Path(__file__).parent.parent / "data" / "knowledge"
RAG_PATH      = Path(__file__).parent.parent / "data" / "rag_db"

# 문서별 출처 라벨 (LLM이 인용할 때 사용)
SOURCE_LABELS = {
    "samsung_car_terms.pdf": "개인용 자동차보험 표준약관",
    "knia_surcharge.pdf":    "손해보험협회 보험료 할증기준 안내",
}

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk(text: str, size: int, overlap: int):
    text = _clean(text)
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build():
    pdfs = sorted(KNOWLEDGE_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"❌ {KNOWLEDGE_DIR} 에 PDF가 없습니다.")
        return

    documents, metadatas, ids = [], [], []
    cid = 0

    for pdf in pdfs:
        label = SOURCE_LABELS.get(pdf.name, pdf.stem)
        print(f"📄 {pdf.name} ({label}) 파싱 중...")
        reader = PdfReader(str(pdf))

        for pno, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            for chunk in _chunk(raw, CHUNK_SIZE, CHUNK_OVERLAP):
                if len(chunk) < 40:       # 너무 짧은 조각 스킵
                    continue
                documents.append(chunk)
                metadatas.append({"source": label, "file": pdf.name, "page": pno})
                ids.append(f"kb_{cid}")
                cid += 1

        print(f"   → 누적 청크 {len(documents)}개")

    print(f"\n🧠 임베딩 + ChromaDB 저장 중 (총 {len(documents)}청크, 수 분 소요)...")
    ef = SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    client = chromadb.PersistentClient(path=str(RAG_PATH))

    try:
        client.delete_collection("knowledge")
    except Exception:
        pass
    col = client.create_collection("knowledge", embedding_function=ef)

    batch = 300
    for start in range(0, len(documents), batch):
        col.add(
            documents=documents[start:start+batch],
            metadatas=metadatas[start:start+batch],
            ids=ids[start:start+batch],
        )
        print(f"  {min(start+batch, len(documents))}/{len(documents)} 저장...")

    print(f"\n🎉 지식 인덱스 빌드 완료 → {RAG_PATH}")
    print(f"   총 {col.count()}청크 저장됨")


if __name__ == "__main__":
    build()
