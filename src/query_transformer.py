import re
from typing import List


class QueryTransformer:

    def __init__(self):

        print("QueryTransformer initialized (deterministic mode)")

    def multi_query(self, original_query: str, num_queries: int = 3) -> List[str]:
        if not original_query or not original_query.strip():
            return []

        base = original_query.strip()


        candidates = [
            base,
            f"Explain {base}",
            f"What are the rules related to {base}?",
            f"Policy regarding {base}",
            f"Company guidelines for {base}",
        ]


        seen = set()
        unique = []
        for q in candidates:
            key = q.lower().strip()
            if key not in seen:
                unique.append(q)
                seen.add(key)

        return unique[:num_queries]

    def hyde(self, query: str) -> str:
        if not query or not query.strip():
            return ""


        return (
            f"This query refers to official organizational policies, rules, "
            f"and documented procedures related to {query}. "
            f"The answer is expected to be found in policy or HR documentation."
        )

    def step_back(self, query: str) -> str:
        if not query or not query.strip():
            return ""

        clean = re.sub(r"[^a-zA-Z0-9 ]", "", query).strip()

        return f"What are the general organizational policies related to {clean}?"
