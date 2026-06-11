"""Vendored helpers from seqlbtoolkit 0.0.9 (text.py, data.py).

Only the functions chempp's parse/save path actually uses are copied here --
verbatim, extracted programmatically from the published wheel -- so the parser
produces byte-identical output without pulling in seqlbtoolkit's heavy
torch/transformers dependency tree.

Source: https://pypi.org/project/SeqLbToolkit/0.0.9/ (MIT-licensed).
"""
import copy
from typing import List, Optional

import regex


# region from seqlbtoolkit/text.py
def format_text(text, remove_ref: Optional[bool] = False, remove_emoj: Optional[bool] = False):
    """
    Normalize text and transform some unicode characters into ascii

    Parameters
    ----------
    text: input text string
    remove_ref: whether remove reference tokens such as [1].
    remove_emoj: whether remove emojis from the text

    Returns
    -------
    normalized text string
    """

    # deal with interpuncts
    interpunct = r'[\u00B7\u02D1\u0387\u05BC\u16EB\u2022\u2027\u2218\u2219\u22C5\u23FA' \
                 r'\u25CF\u25E6\u26AB\u2981\u2E30\u2E31\u2E33\u30FB\uA78F\uFF65]'
    text = regex.sub(interpunct, ' ', text)

    # deal with invisible characters
    text = regex.sub(r'[\u2060-\u2061|\u200b]', '', text)

    # deal with bullets
    bullets = r'[\u2022\u2023\u2043\u204C\u204D\u2219\u25CB\u25D8\u25E6' \
              r'\u2619\u2765\u2767\u29BE\u29BF]'
    text = regex.sub(bullets, ' ', text)

    # deal with quotation marks
    s_quotation = r'[\u2018-\u201B\u2039\u203A]'
    d_quotation = r'[\u00AB\u00BB\u201C\u201D\u201E]'
    text = regex.sub(s_quotation, "'", text)
    text = regex.sub(d_quotation, '"', text)

    # deal with overlay tilde
    tilde = r'[\u0303\u223C\u224B\u02DC\u02F7\u223D\u0360\u0334\u0330\u033E' \
            r'\u1DEC\uFE29\uFE2A\uFE22\uFE23]'
    text = regex.sub(tilde, '~', text)

    # deal with overlay not tilde
    not_tilde = r'[\u034A]'
    text = regex.sub(not_tilde, '≁', text)

    text = remove_combining_marks(text)

    # deal with "/"
    text = regex.sub(r'[ ]?/[ ]?', '/', text)

    # deal with invisible Soft Hyphen
    text = regex.sub(r'[\u00ad]', ' ', text)
    # deal with dash/hyphen
    text = regex.sub(r'[\p{Pd}]', '-', text)
    text = regex.sub(r'[\u2212\uf8ff\uf8fe\ue5f8]', '-', text)

    # deal with `=`
    text = regex.sub(r'[﹦＝]', '=', text)
    text = regex.sub(r"([^ ~≈=≅≤≥⩽⩾<>])([~≈=≅≤≥⩽⩾<>]+)([^ ~≈=≅≤≥⩽⩾<>])", r"\g<1> \g<2> \g<3>", text)

    # convert user-defined characters to dash
    text = regex.sub(r'[\uE000-\uF8FF]', '-', text)

    # deal with repeated comma and period
    text = regex.sub(r'[.]+( *[,.])+', r'..', text)

    # deal with white spaces and \n
    text = regex.sub(r'[\p{Z}]', ' ', text)
    text = regex.sub(r'([ \t]+)?[\r\n]([ \t]+)?', ' ', text)
    text = regex.sub(r'\n+', ' ', text)
    text = regex.sub(r'([ ]{2,})', ' ', text)

    if remove_ref:
        text = remove_reference(text)

    if remove_emoj:
        text = remove_emojis(text)

    text = text.strip()
    return text

def remove_emojis(text):
    """
    Remove emojis in the text
    """
    emoj = regex.compile("["
                         u"\U0001F600-\U0001F64F"  # emoticons
                         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                         u"\U0001F680-\U0001F6FF"  # transport & map symbols
                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                         u"\U00002500-\U00002BEF"  # chinese char
                         u"\U00002702-\U000027B0"
                         u"\U00002702-\U000027B0"
                         u"\U000024C2-\U0001F251"
                         u"\U0001f926-\U0001f937"
                         u"\U00010000-\U0010ffff"
                         u"\u2640-\u2642"
                         u"\u2600-\u2B55"
                         u"\u200d"
                         u"\u23cf"
                         u"\u23e9"
                         u"\u231a"
                         u"\ufe0f"  # dingbats
                         u"\u3030]+", regex.UNICODE)
    return regex.sub(emoj, '', text)

def remove_reference(text: str):
    """
    Remove reference tokens such as [1] from text
    """
    # deal with in-sentence references
    text = regex.sub(r' *\[[0-9-, ]+][,-]*', r'', text)
    # deal with after-sentence references
    text = regex.sub(r'([A-Za-zα-ωΑ-Ω\p{Pe}\'"\u2018-\u201d]+[\d-]*)([.])( |[\d]+)([-, ]*[\d]*)*'
                     r'([ ]+[A-Z\dα-ωΑ-Ω\p{Ps}\'"\u2018-\u201d]|$)', r'\g<1>\g<2>\g<5>', text)

    return text

def remove_combining_marks(text: str):
    """
    Remove combining marks (with unicode 0300-036f)

    Parameters
    ----------
    text: input string text

    Returns
    -------
    string text
    """
    # deal with interpuncts
    diacritics = r'[\u0300-\u036F]'
    text = regex.sub(diacritics, '', text)

    return text

def substring_mapping(text: str, mapping_dict: dict):
    """
    Map substrings in the input string according to the dict

    Parameters
    ----------
    text: input string
    mapping_dict: the mapping dictionary

    Returns
    -------
    string text
    """
    rep = dict((regex.escape(k), v) for k, v in mapping_dict.items())
    pattern = regex.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[regex.escape(m.group(0))], text)
    return text

# endregion


# region from seqlbtoolkit/data.py
def sort_tuples_by_element_idx(tups: List[tuple],
                               idx: Optional[int] = 0,
                               reverse: Optional[bool] = False) -> List[tuple]:
    """
    Function to sort the list of tuples by the second item

    Parameters
    ----------
    tups: a list of tuples to sort
    idx: will sort according to the ith element
    reverse: set True to sort in descending order

    Returns
    -------
    sorted tuples
    """
    tup_ = copy.deepcopy(tups)
    tup_.sort(key=lambda x: x[idx], reverse=reverse)
    return tup_

# endregion
