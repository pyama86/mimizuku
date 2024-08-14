import re

import numpy as np


def anonymize_path(path):
    if path is None:
        return "N/A"
    anonymized_path = re.sub(r"(\.[a-zA-Z0-9]+)\..*", r"\1.*", path)
    anonymized_path = re.sub(r"[0-9]|;.*", "*", anonymized_path)
    return anonymized_path


def safe_transform(encoder, data, unknown_value=-1):
    encoded = []
    for item in data:
        if item in encoder.classes_:
            encoded.append(encoder.transform([item])[0])
        else:
            encoded.append(unknown_value)
    return np.array(encoded)
