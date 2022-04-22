

def blue_func(src):
    blue = src['BLUE'].where(src['BLUE'] != -9999)
    return blue


def green_func(src):
    green = src['GREEN'].where(src['GREEN'] != -9999)
    return green


def red_func(src):
    red = src['RED'].where(src['RED'] != -9999)
    return red


def nir_func(src):
    nir = src['NIR1'].where(src['NIR1'] != -9999)
    return nir


def swir1_func(src):
    swir1 = src['SWIR1'].where(src['SWIR1'] != -9999)
    return swir1


def swir2_func(src):
    swir2 = src['SWIR2'].where(src['SWIR2'] != -9999)
    return swir2