def calculate_insurance_impact(repair_cost: int, annual_premium: int) -> dict:
    """
    수리비와 현재 연간 보험료를 입력받아
    보험처리 시 손익을 계산합니다.

    국내 자동차보험 할증 기준 (근사치):
      - 50만원 미만  → 1년 10% 할증
      - 50~150만원   → 2년 10% 할증
      - 150만원 이상 → 3년 15% 할증
    """
    if repair_cost < 500_000:
        rate  = 0.10
        years = 1
    elif repair_cost < 1_500_000:
        rate  = 0.10
        years = 2
    else:
        rate  = 0.15
        years = 3

    increase_per_year = int(annual_premium * rate)
    total_increase    = increase_per_year * years

    return {
        "recommendation":    "self" if repair_cost <= total_increase else "insurance",
        "increase_per_year": increase_per_year,
        "total_increase":    total_increase,
        "surcharge_rate":    rate,
        "surcharge_years":   years,
    }
