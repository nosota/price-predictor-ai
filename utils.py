import math

def round_to_nearest_10(number):
    if math.isnan(number):
        return number
    else:
        return round(number / 10) * 10
