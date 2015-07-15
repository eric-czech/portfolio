__author__ = 'eczech'

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
import numpy as np


def _rgb_to_lab(rgb):
    orgb = sRGBColor(rgb[0], rgb[1], rgb[2])
    orgb.is_upscaled = False
    r = convert_color(orgb, LabColor).get_value_tuple()
    if not np.all(np.isfinite(r)):
        raise ValueError('Found non finite values in input rgb tuple {} '.format(rgb))
    return r

def rgb_to_lab(img_rgb):
    return np.apply_along_axis(_rgb_to_lab, 2, img_rgb)

def _lab_to_rgb(lab):
    olab = LabColor(lab[0], lab[1], lab[2])
    r = convert_color(olab, sRGBColor).get_value_tuple()
    if not np.all(np.isfinite(r)):
        raise ValueError('Found non finite values in input lab tuple {} '.format(lab))
    return r

def lab_to_rgb(img_lab):
    return np.apply_along_axis(_lab_to_rgb, 2, img_lab)
