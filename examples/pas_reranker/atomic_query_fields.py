"""Infer which PAS atomic requirements are present from query text.

This is deliberately a conservative inference-time helper.  It does not read
the dense HCR/query-attribute assets: those assets are valid for constructing
training labels, but would be ground-truth metadata if used to decide which
atomic prompts to issue at deployment.
"""

from __future__ import annotations

import re


ATOMIC_FIELDS = (
    "top_outer_color",
    "top_outer_type",
    "bottom_color",
    "bottom_type",
    "shoe_color",
    "shoe_type",
    "viewpoint",
    "accessory_subset",
)

_TOP_TYPES = (
    "cropped jacket",
    "knee length coat",
    "knee-length coat",
    "t-shirt",
    "t shirt",
    "camisole",
    "hoodie",
    "jacket",
    "coat",
    "robe",
    "shirt",
    "sweater",
    "vest",
)
# Some PAS captions deliberately specify only a generic upper garment, e.g.
# ``purple top``.  In that case color is an active requirement but ``top`` is
# not a canonical type value and must not create a top-type probe.
_GENERIC_TOP_TYPES = ("top",)
_BOTTOM_TYPES = ("leggings", "shorts", "skirt", "jeans", "pants", "dress")
_SHOE_TYPES = (
    "flip flops",
    "high heels",
    "barefoot",
    "sneakers",
    "sandals",
    "boots",
)
_VIEWPOINTS = (
    "front view",
    "back view",
    "side view",
    "front facing",
    "back facing",
    "from the front",
    "from the back",
    "from the side",
)

# The PAS captions use compositional color names ("dusty rose", "navy blue",
# etc.).  For activation we only need to know that a color modifies a garment;
# exact color equality remains the model's job.
# These are the actual scalar values supported by PAS ``attribute_vocab.json``.
# Adjectives such as ``violet``, ``tan``, and ``dusty rose`` occur in captions
# but map to ``__missing__`` in the ranking labels; treating them as active
# constraints creates systematic false vetoes.  Compound descriptions remain
# supported whenever they contain a canonical token (for example ``navy blue``
# and ``charcoal black``).
_GARMENT_COLORS = {
    "beige",
    "black",
    "blue",
    "brown",
    "camouflage",
    "green",
    "grey",
    "gray",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}
_SHOE_COLORS = _GARMENT_COLORS - {"camouflage"}

_FIELD_READABLE = {
    "top_outer_color": "upper-body clothing color",
    "top_outer_type": "upper-body clothing type",
    "bottom_color": "lower-body clothing color",
    "bottom_type": "lower-body clothing type",
    "shoe_color": "shoe color",
    "shoe_type": "shoe type",
    "viewpoint": "viewpoint",
    "accessory_subset": "accessories",
}
_COLOR_CANONICAL = {
    "beige": "beige", "black": "black", "blue": "blue", "brown": "brown",
    "camouflage": "camouflage", "green": "green", "grey": "grey", "gray": "grey",
    "orange": "orange", "pink": "pink", "purple": "purple", "red": "red",
    "white": "white", "yellow": "yellow",
}

_ACCESSORY_RE = re.compile(
    r"\b(?:bag|bags|backpack|backpacks|purse|purses|handbag|handbags|"
    r"lanyard|lanyards|scarf|scarves|glasses|spectacles|sunglasses|mask|"
    r"hijab|headscarf|hat|hats|beanie|cap|umbrella|watch|watches|glove|"
    r"gloves|briefcase|suitcase|phone|camera|badge|necklace|bracelet|belt|"
    r"earphones|headphones|headband|tie|helmet|strap|clutch|folder|basket|"
    r"bottle|book|books|box|boxes)\b"
)


def _normalize(text: str) -> str:
    # Treat hyphenated PAS spellings (``t-shirt``, ``knee-length``) as the
    # same lexical forms as the space-separated vocabulary.
    return re.sub(r"\s+", " ", text.casefold().replace("–", "-").replace("-", " ")).strip()


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", text) for p in phrases)


def _spans(text: str, phrases: tuple[str, ...]) -> list[tuple[int, int]]:
    return [
        match.span()
        for phrase in phrases
        for match in re.finditer(
            rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text
        )
    ]


def _color_near_type(
    text: str, types: tuple[str, ...], colors: set[str]
) -> bool:
    """Whether a color word occurs in the local phrase around a garment."""

    tokens = re.findall(r"[a-z]+(?:-[a-z]+)?", text)
    color_indices = {index for index, token in enumerate(tokens) if token in colors}
    # A nearby color belongs to the first following garment phrase.  Without
    # this barrier, ``grey sweater, terracotta skirt`` incorrectly assigns
    # ``grey`` to both the sweater and the skirt simply because the second
    # type lies inside a fixed-width window.
    all_type_spans: list[tuple[int, int]] = []
    for candidate in {*_TOP_TYPES, *_BOTTOM_TYPES, *_SHOE_TYPES}:
        candidate_tokens = candidate.replace("-", " ").split()
        candidate_width = len(candidate_tokens)
        for candidate_begin in range(len(tokens) - candidate_width + 1):
            if tokens[candidate_begin : candidate_begin + candidate_width] == candidate_tokens:
                all_type_spans.append(
                    (candidate_begin, candidate_begin + candidate_width)
                )
    for phrase in types:
        phrase_tokens = phrase.replace("-", " ").split()
        width = len(phrase_tokens)
        for begin in range(len(tokens) - width + 1):
            if tokens[begin : begin + width] == phrase_tokens:
                # PAS captions place the color modifier before the item.  A
                # symmetric window incorrectly borrows the color of the next
                # garment (``tan hoodie and red skirt`` => red hoodie).
                for index in sorted(color_indices, reverse=True):
                    if not 0 < begin - index <= 4:
                        continue
                    if any(index < other_begin < begin for other_begin, _ in all_type_spans):
                        continue
                    return True
    return False


def infer_atomic_fields(query: str) -> tuple[str, ...]:
    """Return active PAS fields using only the natural-language query.

    The order is stable and matches the atomic prompt/training field order.
    An empty result is rejected because the atomic scorer cannot form an AND
    without at least one requirement.
    """

    text = _normalize(query)
    top_type = _contains_any(text, _TOP_TYPES)
    generic_top_color = _color_near_type(
        text, _GENERIC_TOP_TYPES, _GARMENT_COLORS
    )
    bottom_type = _contains_any(text, _BOTTOM_TYPES)
    shoe_type = _contains_any(text, _SHOE_TYPES)
    fields: list[str] = []
    if top_type:
        if _color_near_type(text, _TOP_TYPES, _GARMENT_COLORS):
            fields.append("top_outer_color")
        fields.append("top_outer_type")
    elif generic_top_color:
        fields.append("top_outer_color")
    if bottom_type:
        if _color_near_type(text, _BOTTOM_TYPES, _GARMENT_COLORS):
            fields.append("bottom_color")
        fields.append("bottom_type")
    if shoe_type:
        if _color_near_type(
            text,
            tuple(t for t in _SHOE_TYPES if t != "barefoot"),
            _SHOE_COLORS,
        ):
            fields.append("shoe_color")
        fields.append("shoe_type")
    if _contains_any(text, _VIEWPOINTS):
        fields.append("viewpoint")
    if _ACCESSORY_RE.search(text):
        fields.append("accessory_subset")
    if not fields:
        raise ValueError(f"query has no inferable PAS atomic requirement: {query!r}")
    return tuple(fields)


def _find_type_value(text: str, phrases: tuple[str, ...], allowed: set[str]) -> str | None:
    """Return the longest canonical type phrase occurring in normalized text."""
    for phrase in sorted(phrases, key=len, reverse=True):
        canonical = phrase.replace("-", " ")
        if canonical not in allowed:
            continue
        if re.search(rf"(?<![a-z]){re.escape(canonical)}(?![a-z])", text):
            return canonical
    return None


def _find_preceding_color(
    text: str,
    types: tuple[str, ...],
    colors: set[str],
    *,
    prefer_explicit_before_camouflage: bool = False,
) -> str | None:
    """Find the canonical color immediately modifying a garment type."""
    tokens = re.findall(r"[a-z]+", text)
    type_phrases = sorted((p.replace("-", " ").split() for p in types), key=len, reverse=True)
    # A modifier cannot cross *any* garment boundary.  Restricting this
    # barrier to the requested garment family would turn ``yellow shorts,
    # cream sweater`` into a false yellow sweater requirement.
    all_type_phrases = sorted(
        (p.replace("-", " ").split() for p in (*_TOP_TYPES, *_BOTTOM_TYPES, *_SHOE_TYPES)),
        key=len,
        reverse=True,
    )
    for begin in range(len(tokens)):
        for words in type_phrases:
            if tokens[begin : begin + len(words)] != words:
                continue
            for index in range(begin - 1, max(-1, begin - 5), -1):
                if tokens[index] in colors:
                    if (
                        prefer_explicit_before_camouflage
                        and tokens[index] == "camouflage"
                        and index > 0
                        and tokens[index - 1] in colors - {"camouflage"}
                    ):
                        # In the train-only PAS contract, explicit compounds
                        # such as ``green camouflage pants`` resolve to green
                        # substantially more often than to camouflage.  Keep
                        # v1 unchanged; v2 exposes this majority, query-only
                        # interpretation for an isolated prompt experiment.
                        return _COLOR_CANONICAL[tokens[index - 1]]
                    return _COLOR_CANONICAL[tokens[index]]
                # A different garment type blocks borrowing its modifier.
                if any(
                    tokens[index : index + len(other)] == other
                    for other in all_type_phrases
                ):
                    break
    return None


def infer_atomic_field_values(
    query: str, *, prefer_explicit_before_camouflage: bool = False
) -> dict[str, str]:
    """Infer canonical explicit target values from query text only.

    The returned values match ``attribute_vocab.json`` spelling.  Unknown
    descriptive colors (e.g. ``dusty rose``) intentionally yield no scalar
    value, so an explicit-value scorer can abstain instead of inventing a
    target that was never present in training labels.
    """
    text = _normalize(query)
    result: dict[str, str] = {}
    top_type = _find_type_value(
        text, _TOP_TYPES,
        {"camisole", "cropped jacket", "hoodie", "knee length coat", "robe", "shirt", "sweater", "t shirt", "vest"},
    )
    bottom_type = _find_type_value(text, _BOTTOM_TYPES, {"dress", "jeans", "leggings", "pants", "shorts", "skirt"})
    shoe_type = _find_type_value(text, _SHOE_TYPES, {"barefoot", "boots", "flip flops", "high heels", "sandals", "sneakers"})
    if top_type:
        result["top_outer_type"] = top_type
        color = _find_preceding_color(
            text,
            _TOP_TYPES,
            _GARMENT_COLORS,
            prefer_explicit_before_camouflage=prefer_explicit_before_camouflage,
        )
        if color:
            result["top_outer_color"] = color
    else:
        color = _find_preceding_color(
            text,
            _GENERIC_TOP_TYPES,
            _GARMENT_COLORS,
            prefer_explicit_before_camouflage=prefer_explicit_before_camouflage,
        )
        if color:
            result["top_outer_color"] = color
    if bottom_type:
        result["bottom_type"] = bottom_type
        color = _find_preceding_color(
            text,
            _BOTTOM_TYPES,
            _GARMENT_COLORS,
            prefer_explicit_before_camouflage=prefer_explicit_before_camouflage,
        )
        if color:
            result["bottom_color"] = color
    if shoe_type:
        result["shoe_type"] = shoe_type
        if shoe_type != "barefoot":
            color = _find_preceding_color(
                text,
                tuple(t for t in _SHOE_TYPES if t != "barefoot"),
                _SHOE_COLORS,
                prefer_explicit_before_camouflage=prefer_explicit_before_camouflage,
            )
            if color:
                result["shoe_color"] = color
    for viewpoint in ("front view", "back view", "side view"):
        if re.search(rf"(?<![a-z]){re.escape(viewpoint)}(?![a-z])", text):
            result["viewpoint"] = viewpoint
            break
    if _ACCESSORY_RE.search(text):
        result["accessory_subset"] = "present"
    return result


def explicit_atomic_prompt(field: str, value: str) -> str:
    """Canonical prompt shared by train transformation and inference."""
    readable = _FIELD_READABLE[field]
    return (
        f"<image>\nTarget {readable}: {value}. "
        "Does the image show that target? Ignore every other detail. "
        "Answer with <answer>yes</answer> or <answer>no</answer>."
    )


def canonical_conjunction_prompt(
    query: str, *, prefer_explicit_before_camouflage: bool = False
) -> str:
    """Build the query-only conjunction prompt used by the r192 arm.

    The natural query is deliberately retained verbatim.  The checklist is
    only an explicit, canonical rendering of values that the conservative
    parser can recover from that same query; an unrecognized value is omitted
    rather than guessed.  This function is shared by training-data generation
    and deployment scoring, so scoring never needs HCR metadata.
    """
    query = str(query).strip()
    if not query:
        raise ValueError("canonical conjunction prompt requires a nonempty query")
    values = infer_atomic_field_values(
        query,
        prefer_explicit_before_camouflage=prefer_explicit_before_camouflage,
    )
    # ``accessory_subset=present`` is an applicability marker, not a target
    # value.  The original query remains in the prompt and carries its exact
    # accessory wording, so do not turn this marker into a spurious value.
    checklist = [
        (field, value)
        for field, value in values.items()
        if field != "accessory_subset"
    ]
    lines = [
        "<image>",
        f'Query: "{query}"',
        "Known explicit target checklist (inferred from the query):",
    ]
    if checklist:
        lines.extend(f"- {_FIELD_READABLE[field]}: {value}" for field, value in checklist)
    else:
        lines.append("- (no canonical field values recovered; use the original query)")
    lines.extend(
        [
            "ALL requirements in the original query must match. The checklist restates only parsed targets and does not replace the original query.",
            "Does the person in the image fully match the query? Answer with <answer>yes</answer> or <answer>no</answer>.",
        ]
    )
    return "\n".join(lines)


_EXCLUSIVE_TYPE_ALTERNATIVES = {
    "bottom_type": {
        "pants": ("jeans", "leggings", "skirt", "shorts", "dress"),
        "jeans": ("pants", "leggings"),
        "leggings": ("pants", "jeans"),
        "skirt": ("dress", "pants", "shorts"),
        "shorts": ("skirt", "pants"),
        "dress": ("skirt", "pants"),
    },
    "top_outer_type": {
        "shirt": ("t shirt", "sweater", "cropped jacket", "knee length coat"),
        "t shirt": ("shirt", "camisole", "sweater", "vest"),
        "camisole": ("t shirt",),
        "sweater": ("shirt", "t shirt", "hoodie"),
        "hoodie": ("sweater", "shirt"),
        "cropped jacket": ("shirt",),
        "knee length coat": ("shirt", "robe", "hoodie"),
        "robe": ("knee length coat",),
        "vest": ("t shirt", "shirt"),
    },
}


def taxonomy_exclusive_conjunction_prompt(query: str) -> str:
    """Canonical conjunction prompt with query-only PAS type disambiguation.

    The extra rule does not infer from the image or from HCR labels.  It is a
    fixed rendering determined solely by type values parsed from ``query`` and
    therefore can be called byte-identically during training and deployment.
    """
    base = canonical_conjunction_prompt(query)
    values = infer_atomic_field_values(str(query))
    rules = []
    for field in ("top_outer_type", "bottom_type"):
        value = values.get(field)
        alternatives = _EXCLUSIVE_TYPE_ALTERNATIVES.get(field, {}).get(value or "")
        if alternatives:
            rules.append(
                f"- {_FIELD_READABLE[field]} {value}: accept exactly {value}; "
                f"reject {', '.join(alternatives)}."
            )
    if not rules:
        return base
    marker = "ALL requirements in the original query must match."
    insertion = (
        "PAS clothing-type labels are mutually exclusive evaluation categories, not hierarchical synonyms.\n"
        + "\n".join(rules)
        + "\n"
    )
    if marker not in base:
        raise RuntimeError("canonical conjunction prompt contract changed")
    return base.replace(marker, insertion + marker, 1)


def label_compatible_conjunction_prompt_from_values(
    query: str,
    values: dict[str, str],
    *,
    has_accessory: bool | None = None,
) -> str:
    """Render the PAS scoring contract from already-canonical field values.

    PAS captions contain fine-grained color words such as ``cream``, ``tan``,
    and ``burgundy`` that are stored as a missing scalar target.  Repeating the
    raw caption as an ordinary all-requirements query makes those words active
    for the VLM while the hard label explicitly ignores them.  This prompt
    states the dataset contract: canonical scalar values are enforced only
    when recovered into the checklist.  The original text remains available
    solely for the open-vocabulary accessory subset, whose labels are exact
    phrases rather than scalar attributes.
    """

    query = str(query).strip()
    if not query:
        raise ValueError("label-compatible conjunction prompt requires a nonempty query")
    if has_accessory is None:
        has_accessory = "accessory_subset" in values
    checklist = [
        (field, value)
        for field, value in values.items()
        if field != "accessory_subset"
    ]
    lines = [
        "<image>",
        "PAS scoring contract:",
        "Check every scalar requirement listed below and no unlisted scalar adjective.",
    ]
    if checklist:
        lines.extend(f"- {_FIELD_READABLE[field]}: {value}" for field, value in checklist)
    else:
        lines.append("- no canonical scalar requirement")
    if has_accessory:
        lines.extend(
            [
                f'Accessory wording source: "{query}"',
                "Also require every explicitly stated carried or worn accessory item.",
            ]
        )
    lines.extend(
        [
            "Ignore descriptive colors or scalar words from the source text when they are absent from the checklist.",
            "Does the person satisfy all scored requirements? Answer with <answer>yes</answer> or <answer>no</answer>.",
        ]
    )
    return "\n".join(lines)


def label_compatible_conjunction_prompt(query: str) -> str:
    """Infer query-only canonical values and render the PAS scoring contract.

    PAS captions contain fine-grained color words such as ``cream``, ``tan``,
    and ``burgundy`` that are stored as a missing scalar target.  Repeating the
    raw caption as an ordinary all-requirements query makes those words active
    for the VLM while the hard label explicitly ignores them.  This prompt
    states the dataset contract: canonical scalar values are enforced only
    when recovered into the checklist.  The original text remains available
    solely for the open-vocabulary accessory subset, whose labels are exact
    phrases rather than scalar attributes.
    """

    query = str(query).strip()
    values = infer_atomic_field_values(query)
    return label_compatible_conjunction_prompt_from_values(query, values)


def label_compatible_conjunction_prompt_v2_from_values(
    query: str, values: dict[str, str]
) -> str:
    """Render the minimal-edit label-compatible prompt from canonical values."""
    query = str(query).strip()
    if not query:
        raise ValueError("label-compatible conjunction prompt requires a nonempty query")
    checklist = [
        (field, value)
        for field, value in values.items()
        if field != "accessory_subset"
    ]
    lines = [
        "<image>",
        f'Query: "{query}"',
        "Canonical scalar requirements used by the PAS labels:",
    ]
    if checklist:
        lines.extend(f"- {_FIELD_READABLE[field]}: {value}" for field, value in checklist)
    else:
        lines.append("- no canonical scalar requirement")
    lines.extend(
        [
            "For clothing colors, clothing types, shoe attributes, and viewpoint, enforce exactly the checklist and ignore unlisted descriptive modifiers in the query.",
            "Still enforce any carried or worn accessory items explicitly stated in the query.",
            "Does the person satisfy all scored requirements? Answer with <answer>yes</answer> or <answer>no</answer>.",
        ]
    )
    return "\n".join(lines)


def label_compatible_conjunction_prompt_v2(query: str) -> str:
    """Infer query-only values for the minimal-edit label-compatible prompt."""

    query = str(query).strip()
    return label_compatible_conjunction_prompt_v2_from_values(
        query, infer_atomic_field_values(query)
    )
