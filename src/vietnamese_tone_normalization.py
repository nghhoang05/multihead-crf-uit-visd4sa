"""
Vietnamese tone-mark placement normalization: "new style" (tone mark on the
phonetically-stressed vowel, e.g. "hòa") -> "old style" ("hoà"). Not Unicode
NFC/NFD -- a purely orthographic convention mismatch, independent of
encoding.

Needed because PhoW2V's training corpus was preprocessed with this exact
table, so looking up a "new style" syllable (the common modern spelling,
including UIT-ViSD4SA's reviews) against PhoW2V's vocabulary needs the same
rewrite first, or oa/oe/uy syllables spuriously miss and fall back to <UNK>.
Table source: VinAIResearch/BARTpho, VietnameseToneNormalization.md.
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
