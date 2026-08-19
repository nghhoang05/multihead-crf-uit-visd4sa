"""
Vietnamese tone-mark placement normalization ("new style" -> "old style"),
e.g. "hoà" -> "hòa" is NOT what this does -- it's the reverse direction:
diphthongs written with the tone mark on the phonetically-stressed vowel
("hòa", "thúy") are rewritten with the tone mark on the old-style/visually
centered vowel ("hoà", "thuý").

This is NOT Unicode NFC/NFD normalization (combining vs precomposed
diacritics) -- it's a purely orthographic convention mismatch between two
common Vietnamese typing/spelling styles, independent of encoding.

Required because PhoW2V's 20GB training corpus was preprocessed with this
exact table before training (see VinAI's PhoW2V README), so looking up a
syllable that came from "new style" text (the overwhelming majority of
modern Vietnamese software output, including our UIT-ViSD4SA reviews)
against PhoW2V's vocabulary needs the same rewrite first, or many oa/oe/uy
syllables will spuriously miss and fall back to <UNK>.

Source: VinAIResearch/BARTpho, VietnameseToneNormalization.md (dict_map
reproduced verbatim -- it is a small, fixed orthographic lookup table, not
prose/creative content).
"""
from __future__ import annotations

DICT_MAP = {
    "òa": "oà", "Òa": "Oà", "ÒA": "OÀ",
    "óa": "oá", "Óa": "Oá", "ÓA": "OÁ",
    "ỏa": "oả", "Ỏa": "Oả", "ỎA": "OẢ",
    "õa": "oã", "Õa": "Oã", "ÕA": "OÃ",
    "ọa": "oạ", "Ọa": "Oạ", "ỌA": "OẠ",
    "òe": "oè", "Òe": "Oè", "ÒE": "OÈ",
    "óe": "oé", "Óe": "Oé", "ÓE": "OÉ",
    "ỏe": "oẻ", "Ỏe": "Oẻ", "ỎE": "OẺ",
    "õe": "oẽ", "Õe": "Oẽ", "ÕE": "OẼ",
    "ọe": "oẹ", "Ọe": "Oẹ", "ỌE": "OẸ",
    "ùy": "uỳ", "Ùy": "Uỳ", "ÙY": "UỲ",
    "úy": "uý", "Úy": "Uý", "ÚY": "UÝ",
    "ủy": "uỷ", "Ủy": "Uỷ", "ỦY": "UỶ",
    "ũy": "uỹ", "Ũy": "Uỹ", "ŨY": "UỸ",
    "ụy": "uỵ", "Ụy": "Uỵ", "ỤY": "UỴ",
}


def normalize_tone(text: str) -> str:
    """Rewrite oa/oe/uy diphthongs from new-style to old-style tone-mark
    placement, matching the preprocessing PhoW2V's corpus went through."""
    for new_style, old_style in DICT_MAP.items():
        text = text.replace(new_style, old_style)
    return text
