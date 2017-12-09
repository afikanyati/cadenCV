import cv2
import numpy as np


def fit(img, templates, width, height, threshold):
    locations = []
    location_count = 0
    for template in templates:
        template = cv2.resize(template,(width, height), interpolation=cv2.INTER_CUBIC)
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        result = np.where(result >= threshold)
        location_count += len(result[0])
        locations += [result]

    return locations
