BUSINESS_TERMS: dict[str, list[str]] = {
    "revenue": ["revenue", "income", "sales", "gmv", "收入", "营收", "销售额"],
    "signup": ["signup", "registration", "new user", "注册", "新用户"],
    "active_user": ["active user", "dau", "mau", "活跃", "活跃用户"],
    "conversion": ["conversion", "convert", "转化", "转化率"],
}


def normalize_business_terms(question: str) -> list[str]:
    lowered = question.lower()
    matched: list[str] = []
    for canonical, aliases in BUSINESS_TERMS.items():
        if any(alias in lowered or alias in question for alias in aliases):
            matched.append(canonical)
    return matched
