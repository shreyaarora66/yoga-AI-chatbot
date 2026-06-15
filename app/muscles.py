"""Canonical muscle names and alias normalization.

The 17 canonical muscles match the ``primaryMuscles`` values in data.json (and the
Muscle-Beginner-Intermediate-Expert CSV). User/LLM phrasing is messy ("delts",
"abs", "legs"), so we normalize to canonical names before keying history and
progression. Group words like "legs"/"arms"/"back" expand to several muscles.
"""

from __future__ import annotations

# Canonical muscles (lowercase, exactly as stored in data.json).
CANONICAL: tuple[str, ...] = (
    "shoulders",
    "abdominals",
    "quadriceps",
    "chest",
    "triceps",
    "biceps",
    "hamstrings",
    "lats",
    "middle back",
    "forearms",
    "calves",
    "glutes",
    "lower back",
    "traps",
    "adductors",
    "neck",
    "abductors",
)

# Single-muscle aliases -> one canonical muscle.
_ALIASES: dict[str, str] = {
    "shoulder": "shoulders",
    "delt": "shoulders",
    "delts": "shoulders",
    "deltoid": "shoulders",
    "deltoids": "shoulders",
    "ab": "abdominals",
    "abs": "abdominals",
    "core": "abdominals",
    "stomach": "abdominals",
    "quad": "quadriceps",
    "quads": "quadriceps",
    "quadricep": "quadriceps",
    "thigh": "quadriceps",
    "thighs": "quadriceps",
    "pec": "chest",
    "pecs": "chest",
    "chest": "chest",
    "tricep": "triceps",
    "bicep": "biceps",
    "hamstring": "hamstrings",
    "hams": "hamstrings",
    "lat": "lats",
    "midback": "middle back",
    "mid back": "middle back",
    "upper back": "middle back",
    "rhomboids": "middle back",
    "forearm": "forearms",
    "calf": "calves",
    "calfs": "calves",
    "glute": "glutes",
    "butt": "glutes",
    "buttocks": "glutes",
    "lowerback": "lower back",
    "low back": "lower back",
    "trap": "traps",
    "trapezius": "traps",
    "adductor": "adductors",
    "abductor": "abductors",
}

# Group words -> several canonical muscles.
_GROUPS: dict[str, tuple[str, ...]] = {
    "legs": ("quadriceps", "hamstrings", "calves", "glutes"),
    "leg": ("quadriceps", "hamstrings", "calves", "glutes"),
    "arms": ("biceps", "triceps", "forearms"),
    "arm": ("biceps", "triceps", "forearms"),
    "back": ("lats", "middle back", "lower back"),
    "full body": ("chest", "lats", "quadriceps", "shoulders", "abdominals"),
    "fullbody": ("chest", "lats", "quadriceps", "shoulders", "abdominals"),
    "upper body": ("chest", "lats", "shoulders", "biceps", "triceps"),
    "lower body": ("quadriceps", "hamstrings", "calves", "glutes"),
}


def normalize_token(name: str) -> list[str]:
    """Map one user term to a list of canonical muscles (empty if unknown)."""
    key = " ".join((name or "").strip().lower().split())
    if not key:
        return []
    if key in _GROUPS:
        return list(_GROUPS[key])
    if key in CANONICAL:
        return [key]
    if key in _ALIASES:
        return [_ALIASES[key]]
    # Try a trailing-s singular/plural fix.
    if key.endswith("s") and key[:-1] in _ALIASES:
        return [_ALIASES[key[:-1]]]
    if not key.endswith("s") and (key + "s") in CANONICAL:
        return [key + "s"]
    return []


def normalize_muscles(names: list[str]) -> list[str]:
    """Expand and de-duplicate a list of user terms into canonical muscles.

    Unknown terms are kept (lowercased) so semantic search can still try them.
    """
    result: list[str] = []
    for name in names or []:
        expanded = normalize_token(name)
        if not expanded:
            cleaned = " ".join((name or "").strip().lower().split())
            expanded = [cleaned] if cleaned else []
        for muscle in expanded:
            if muscle not in result:
                result.append(muscle)
    return result
