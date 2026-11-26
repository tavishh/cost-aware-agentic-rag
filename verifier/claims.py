"""Conservative, evidence-independent QA-to-claim construction for experiments.

Rules depend on question grammar, never dataset IDs, gold answers or scores.
Unrecognized conversions must pass checks; failure is missing, not score zero.
"""
from dataclasses import dataclass
import re

VERSION = "literal_answer_scope_guard_v2"


@dataclass(frozen=True)
class Claim:
    text: str
    strategy: str
    issues: tuple = ()


def words(text):
    return re.findall(r"\w+", text.casefold())


def literal_answer(answer):
    return answer.strip().rstrip(".!?").strip()


def template_claim(question, answer):
    q, a = question.strip().rstrip("?"), literal_answer(answer)
    if not q or not a:
        return None
    patterns = [
        (r"where (was|were|is|are) (.+?) (filmed|shot|located|set|born|held)",
         lambda m: f"{m[2]} {m[1]} {m[3]} in {a}."),
        (r"how much (?:does|did) it cost to (.+)", lambda m: f"It costs {a} to {m[1]}."),
        (r"how long (?:does|did) it take to (.+)", lambda m: f"It takes {a} to {m[1]}."),
        (r"who (plays|played|portrays|portrayed|sings|sang|wrote|founded|created|invented|designed|directed|discovered|led) (.+)",
         lambda m: f"{a} {m[1]} {m[2]}."),
        (r"who (is|was) (the .+)", lambda m: f"{a} {m[1]} {m[2]}."),
        (r"when (was|were) (.+?) (released|born|founded|created|built|published|established)",
         lambda m: f"{m[2]} {m[1]} {m[3]} on {a}."),
        (r"who (is|was) (.+?) supposed to be (.+)",
         lambda m: f"{m[2]} {m[1]} supposed to be {a} {m[3]}."),
        (r"name (the .+)", lambda m: f"{a} is {m[1]}."),
    ]
    for pattern, build in patterns:
        match = re.fullmatch(pattern, q, flags=re.I)
        if match:
            return Claim(build(match), "grammar_template")
    # Compare only when the draft unambiguously identifies one offered option.
    match = re.fullmatch(r"what came out first (.+?) or (.+)", q, re.I)
    if match:
        opts = [match[1], match[2]]
        norm = lambda s: " ".join(words(s))
        matches = [i for i, option in enumerate(opts) if norm(option) in norm(a)]
        if len(matches) == 1:
            return Claim(f"{a} was released before {opts[1-matches[0]]}.", "comparison_template")
    return None


def restore_answer(answer, generated):
    """Replace the converter's answer span with the original literal answer.

    Token order and multiplicity must match; a set-of-words check is insufficient
    for values such as 10,000-30,000 or multi-entity answers.
    """
    tokens = words(literal_answer(answer))
    if not tokens:
        return None
    pattern = r"(?<!\w)" + r"\W+".join(re.escape(t) for t in tokens) + r"(?!\w)"
    matches = list(re.finditer(pattern, generated, re.I))
    if len(matches) != 1:
        return None
    m = matches[0]
    # Include currency symbols/signs before numeric spans to avoid doubling them.
    start = m.start()
    if literal_answer(answer).startswith(("$", "£", "€", "+", "-")):
        prefix = re.search(r"[$£€+-]\s*$", generated[:start])
        if prefix:
            start = prefix.start()
    return generated[:start] + literal_answer(answer) + generated[m.end():]


def scope_issues(question, claim):
    issues = []
    q, h = set(words(question)), set(words(claim))
    constraints = {
        "first": {"first", "earliest", "before", "original"},
        "last": {"last", "latest", "after"},
        "largest": {"largest", "biggest"}, "smallest": {"smallest"},
        "complex": {"complex"}, "most": {"most"},
        "filmed": {"filmed", "shot"}, "shot": {"shot", "filmed"},
        "not": {"not", "never", "no"}, "never": {"never", "not"},
    }
    for word, equivalents in constraints.items():
        if word in q and not h.intersection(equivalents):
            issues.append(f"lost_question_constraint:{word}")
    for number in re.findall(r"\d[\d,]*(?:\.\d+)?", question):
        if not re.search(r"(?<!\d)" + re.escape(number) + r"(?!\d)", claim):
            issues.append(f"lost_question_number:{number}")
    return tuple(issues)


def build_claim(question, answer, generated=None):
    result = template_claim(question, answer)
    if result is not None:
        # Rules copy the entire relation and literal answer, or both comparison options.
        return result
    restored = restore_answer(answer, generated or "")
    if restored is None:
        return Claim(generated or "", "qa2d_rejected", ("answer_span_missing_or_ambiguous",))
    issues = scope_issues(question, restored)
    return Claim(restored, "qa2d_literal_restoration", issues)
