from singing_girl import sing
import re


_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_number_re = re.compile(r"[0-9]+")


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " punto ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dólares"  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dólar" if dollars == 1 else "dólares"
        cent_unit = "centavo" if cents == 1 else "centavos"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dólar" if dollars == 1 else "dólares"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "centavo" if cents == 1 else "centavos"
        return "%s %s" % (cents, cent_unit)
    else:
        return "cero dolares"
    

def _expand_number(m):
    num = int(m.group(0))
    return sing(num)


def normalize_numbers_es(text):
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r"\1 libras", text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


if __name__ == "__main__":
    print(sing(12))
    # raw_data/LJSpeech/LJSpeech/LJ001-0077_src.lab
    en = "in england about this time an attempt was made notably by caslon who started business in london as a typefounder in seventeen twenty"
    from deep_translator import GoogleTranslator
    translation = GoogleTranslator(source='auto', target='es').translate(en)
    print(translation)
    translation = normalize_numbers_es(translation)
    print(translation)
    pass
