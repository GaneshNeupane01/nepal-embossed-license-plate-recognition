import re
import string

LATIN_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
PROVINCES = [
    "BAGMATI","KOSHI","GANDAKI","KARNALI",
    "LUMBINI","SUDURPASHCHIM","MADHESH",
    "STATE1","STATE2","STATE3","STATE4","STATE5","STATE6","STATE7"
]

CONFUSION_MAP = {
    '0': ['O', 'Q', 'D','G'],
    '1': ['I', 'L', '|'],
    '2': ['Z'],
    '5': ['S'],
    '6': ['G'],
    '8': ['B'],
    'A': ['4'],
    'B': ['8','3'],
    'O': ['0'],
    'S': ['5']
}

def clean_text(text: str) -> str:
    ALLOWED_CHARS = string.ascii_uppercase + string.digits
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    for p in PROVINCES:
        text = text.replace(p, "")
    text = text.replace("NEP", "")
    text = "".join([c for c in text if c in ALLOWED_CHARS])

    return text

def correct_plate(text: str):
    if len(text) >= 7:
        text = text[-7:]

    chars = list(text)

    # First 3 letters
    for i in range(min(3, len(chars))):
        if chars[i].isdigit():
            for k, v in CONFUSION_MAP.items():
                if k.isalpha() and chars[i] in v:
                    chars[i] = k

    # Last 4 digits
    for i in range(len(chars)-4, len(chars)):
        if chars[i].isalpha():
            for k, v in CONFUSION_MAP.items():
                if k.isdigit() and chars[i] in v:
                    chars[i] = k

    return "".join(chars)
