from pathlib import Path

RAG_PATH = Path(__file__).parent.parent / "data" / "rag_db"

_client = None
_ef     = None
_cols   = {}   # 컬렉션 캐시 {name: collection}

_DAMAGE_TO_ACTION = {
    "Scratched": "도장",
    "Breakage":  "교환",
    "Crushed":   "판금",
    "Separated": "탈착",
}


def _get_collection(name: str):
    """ChromaDB 컬렉션 로드 (없으면 None). 클라이언트·임베딩 함수는 재사용."""
    global _client, _ef
    if name in _cols:
        return _cols[name]
    if not RAG_PATH.exists():
        return None
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        if _ef is None:
            _ef = SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            )
        if _client is None:
            _client = chromadb.PersistentClient(path=str(RAG_PATH))
        col = _client.get_collection(name, embedding_function=_ef)
    except Exception:
        _cols[name] = None
        return None
    _cols[name] = col
    return col


def retrieve_similar_cases(damages: list, n_results: int = 3) -> list:
    """감지된 손상으로 유사 수리 사례 검색 (repair_cases 컬렉션)."""
    col = _get_collection("repair_cases")
    if col is None:
        return []

    cases, seen = [], set()
    for dmg in damages:
        part   = dmg.get("part", "")
        action = _DAMAGE_TO_ACTION.get(dmg.get("type", ""), "수리")
        query  = f"부위: {part} | 작업: {action}"
        try:
            res = col.query(query_texts=[query], n_results=n_results)
        except Exception:
            continue
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            key = (meta["part"], meta["action"])
            if key in seen:
                continue
            seen.add(key)
            cases.append({
                "text":   doc,
                "part":   meta["part"],
                "action": meta["action"],
                "car":    meta.get("car", ""),
                "total":  meta.get("total", 0),
            })
    return cases


def retrieve_knowledge(queries: list, n_results: int = 3) -> list:
    """
    보험 약관·할증·법률 지식 문서 검색 (knowledge 컬렉션).
    queries: 검색할 질의 문자열 리스트 (예: ["보험료 할증 기준", "대물배상 보상 범위"])
    반환: [{"source": 출처, "page": 페이지, "text": 조항}] (중복 제거).
    """
    col = _get_collection("knowledge")
    if col is None:
        return []

    hits, seen = [], set()
    for q in queries:
        try:
            res = col.query(query_texts=[q], n_results=n_results)
        except Exception:
            continue
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            key = (meta.get("file"), meta.get("page"), doc[:40])
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "source": meta.get("source", "문서"),
                "page":   meta.get("page", 0),
                "text":   doc,
            })
    return hits
