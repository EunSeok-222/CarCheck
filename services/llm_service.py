import os
from groq import Groq
from services.rag_service import retrieve_similar_cases, retrieve_knowledge

MODEL   = "llama-3.3-70b-versatile"
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client

_SYSTEM_BASE = """당신은 자동차 손상 보험 상담 전문 AI입니다.
오직 아래 주제에만 답변하세요:
- 자동차 보험 (가입, 보상, 할증, 할인, 약관)
- 차량 손상 수리 및 비용
- 보험 처리 vs 자비 처리 비교
- 자동차 사고 관련 절차

위 주제와 무관한 질문(음식, 날씨, 일상 대화 등)은 반드시 다음 문장으로만 답변하세요:
"죄송합니다, 저는 자동차 보험 및 차량 손상 관련 질문만 답변드릴 수 있어요. 보험 관련 궁금한 점이 있으시면 질문해주세요! 😊"

데이터 출처:
- 삼성화재 개인용 자동차보험 약관 (2025.08.16 개정)
- 손해보험협회 자동차보험료 할인·할증 기준 (2024년)

답변 규칙:
1. 위 문서에서 관련 내용을 찾아 근거를 인용 (예: "삼성화재 약관 기준에 따르면...")
2. 문서에 없는 내용은 "일반적으로..." 로 구분
3. 마크다운 형식, 간결하게
4. 끝에 항상: ⚠️ 실제 처리 시 가입 보험사 약관을 직접 확인하세요."""

ACTION_KO = {
    "coating":     "도색",
    "exchange":    "교체",
    "repair":      "수리",
    "sheet_metal": "판금",
}

_DAMAGE_ACTION_MAP = {
    "Scratched": "도색",
    "Breakage":  "교체",
    "Crushed":   "판금",
    "Separated": "수리",
}


def _build_prompt(damage_result: dict, repair_cost: dict, insurance_result: dict,
                  similar_cases: list, knowledge: list) -> str:
    damages = damage_result["damages"]
    total   = repair_cost["total"]
    ins     = insurance_result

    damage_lines = "\n".join(
        f"- {d['part']}: {d['type_ko']} (신뢰도 {d['confidence']:.0%})"
        for d in damages
    )
    repair_lines = "\n".join(
        f"- {item['part']} {ACTION_KO.get(item['action'], item['action'])}: {item['cost']:,}원"
        for item in repair_cost["breakdown"]
    )
    recommend = "자비처리" if ins["recommendation"] == "self" else "보험처리"
    saving    = abs(ins["total_increase"] - total)

    rag_section = ""
    if similar_cases:
        case_lines = "\n".join(f"- {c['text']}" for c in similar_cases[:5])
        rag_section = f"""
**실제 수리 사례 (유사 견적서 {len(similar_cases)}건):**
{case_lines}
"""

    knowledge_section = ""
    if knowledge:
        kb_lines = "\n".join(
            f"- [{k['source']} p.{k['page']}] {k['text']}" for k in knowledge[:5]
        )
        knowledge_section = f"""
**관련 규정·약관 근거 (검색된 공식 문서):**
{kb_lines}
"""

    return f"""당신은 자동차 사고 수리비 전문 AI 분석가입니다.
아래 분석 데이터와 공식 규정 문서를 바탕으로 차주에게 친절하고 명확한 한국어 리포트를 작성하세요.
마크다운 형식으로 작성하되, 핵심 내용만 간결하게 설명하세요.

## 입력 데이터

**감지된 손상:**
{damage_lines}

**예상 수리 내역:**
{repair_lines}
예상 수리비 합계: {total:,}원
{rag_section}{knowledge_section}
**보험 정보:**
- 연간 보험료: {ins['increase_per_year'] * 10:,}원 (추정)
- 보험 처리 시 할증: {ins['surcharge_years']}년간 연 {ins['surcharge_rate']*100:.0f}% → 총 {ins['total_increase']:,}원 인상
- 자비 처리 비용: {total:,}원
- 추천: {recommend} ({saving:,}원 유리)

## 작성 지침
1. 손상 상태를 차주 눈높이에서 쉽게 설명 (기술 용어 최소화)
2. 실제 수리 사례가 있으면 근거로 활용해 비용 범위를 설명
3. 보험처리 vs 자비처리 판단 시, 위 '규정·약관 근거'를 인용하며 설명 (예: "○○약관 기준에 따르면...")
4. 최종 추천 이유를 2~3문장으로
5. 주의사항 1가지 (AI 예측 한계 + 실제 약관·개별 계약 확인 필요 언급)
""".strip()


def _build_kb_queries(damage_result: dict) -> list:
    """감지된 손상 기반으로 규정 검색 질의 생성."""
    queries = [
        "자동차보험 보험료 할증 기준",
        "물적사고 할증기준금액 대물 자기차량손해",
        "자기부담금 보상 범위",
    ]
    for d in damage_result.get("damages", []):
        queries.append(f"{d.get('part','')} {d.get('type_ko','')} 수리 보상")
    return queries


def answer_question(user_query: str, chat_history: list, analysis_context: str = "") -> str:
    """
    보험 관련 추가 질문에 RAG + ollama.chat()으로 답변.
    chat_history: [{"role": "user"|"assistant", "content": "..."}]
    """
    # RAG: 질문과 관련된 약관·할증 규정 검색
    knowledge = retrieve_knowledge([user_query], n_results=4)
    rag_text = ""
    if knowledge:
        rag_text = "\n\n[검색된 관련 규정]\n" + "\n".join(
            f"- [{k['source']} p.{k['page']}] {k['text']}" for k in knowledge
        )

    system_content = _SYSTEM_BASE + rag_text
    if analysis_context:
        system_content += f"\n\n[이번 상담 차량 분석 결과]\n{analysis_context}"

    messages = [{"role": "system", "content": system_content}]
    # 최근 6턴만 유지 (컨텍스트 과부하 방지)
    messages += chat_history[-12:]
    messages.append({"role": "user", "content": user_query})

    try:
        resp = _get_client().chat.completions.create(
            model=MODEL, messages=messages,
            temperature=0.1, max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"죄송합니다, 답변 생성 중 오류가 발생했습니다. ({e})"


def generate_report(damage_result: dict, repair_cost: dict, insurance_result: dict) -> str:
    similar_cases = retrieve_similar_cases(damage_result.get("damages", []))
    knowledge     = retrieve_knowledge(_build_kb_queries(damage_result))
    prompt = _build_prompt(damage_result, repair_cost, insurance_result,
                           similar_cases, knowledge)
    try:
        resp = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=700,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return _fallback_report(damage_result, repair_cost, insurance_result, error=str(e))


def _fallback_report(damage_result, repair_cost, insurance_result, error="") -> str:
    damages   = damage_result["damages"]
    total     = repair_cost["total"]
    ins       = insurance_result
    recommend = "자비처리" if ins["recommendation"] == "self" else "보험처리"
    saving    = abs(ins["total_increase"] - total)

    damage_lines = "\n".join(
        f"- **{d['part']}**: {d['type_ko']} (신뢰도 {d['confidence']:.0%})"
        for d in damages
    )
    repair_lines = "\n".join(
        f"- {item['part']} {ACTION_KO.get(item['action'], item['action'])}: {item['cost']:,}원"
        for item in repair_cost["breakdown"]
    )

    note = f"\n\n> ⚠️ LLM 연결 오류 (템플릿 출력): {error}" if error else ""

    return f"""## 🔍 손상 분석 요약

총 **{len(damages)}곳**의 손상이 감지되었습니다.

{damage_lines}

---

## 🔧 예상 수리 내역

{repair_lines}

**예상 수리비 합계: {total:,}원**

---

## 💡 처리 방법 추천

| 항목 | 보험처리 | 자비처리 |
|------|---------|---------|
| 수리비 부담 | 0원 | {total:,}원 |
| {ins['surcharge_years']}년간 보험료 인상 | +{ins['total_increase']:,}원 | 0원 |
| **실질 부담 합계** | **{ins['total_increase']:,}원** | **{total:,}원** |

**→ {recommend}가 약 {saving:,}원 더 유리합니다.**

> ⚠️ 본 분석은 AI 예측값이며 실제 견적과 다를 수 있습니다.{note}
"""
