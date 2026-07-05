
BIOMARKER_NORMALIZATION = {

    # ESR1
    "esr1 activating variants":
        "esr1 oncogenic variants",

    # ERBB2
    "erbb2 activating variants":
        "erbb2 oncogenic variants",
}


def normalize_biomarker(text):

    text = text.lower().strip()

    return BIOMARKER_NORMALIZATION.get(text, text)
