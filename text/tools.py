from text.ipadict import db

def split_with_tie_bar(text):
    """
    Splits phonetic sentence into list of phonetic words.
    Maintaines Affricates by grouping them with the tie bar.
    Skips characters not in the ipa dictionary (such as diacritics).
    """
    result = []
    i = 0
    while i < len(text):
        if i + 2 < len(text) and text[i + 1] == '͡':
            # Group the character with the tie bar and the following character
            if text[i:i+3] in db.keys():
                result.append(text[i:i + 3])
            i += 3
        else:
            # Just append the single character
            if text[i] in db.keys():
                result.append(text[i])
            i += 1
    return result