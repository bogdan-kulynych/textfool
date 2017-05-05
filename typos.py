# Based on https://github.com/Woorank/tipo/

KEY_MISHITS_MAP = {
  '1': [ '2', 'q' ],
  '2': [ '1', 'q', 'w', '3' ],
  '3': [ '2', 'w', 'e', '4' ],
  '4': [ '3', 'e', 'r', '5' ],
  '5': [ '4', 'r', 't', '6' ],
  '6': [ '5', 't', 'y', '7' ],
  '7': [ '6', 'y', 'u', '8' ],
  '8': [ '7', 'u', 'i', '9' ],
  '9': [ '8', 'i', 'o', '0' ],
  '0': [ '9', 'o', 'p', '-' ],
  '-': [ '0', 'p' ],
  'q': [ '1', '2', 'w', 'a' ],
  'w': [ 'q', 'a', 's', 'e', '3', '2' ],
  'e': [ 'w', 's', 'd', 'r', '4', '3' ],
  'r': [ 'e', 'd', 'f', 't', '5', '4' ],
  't': [ 'r', 'f', 'g', 'y', '6', '5' ],
  'y': [ 't', 'g', 'h', 'u', '7', '6' ],
  'u': [ 'y', 'h', 'j', 'i', '8', '7' ],
  'i': [ 'u', 'j', 'k', 'o', '9', '8' ],
  'o': [ 'i', 'k', 'l', 'p', '0', '9' ],
  'p': [ 'o', 'l', '-', '0' ],
  'a': [ 'z', 's', 'w', 'q' ],
  's': [ 'a', 'z', 'x', 'd', 'e', 'w' ],
  'd': [ 's', 'x', 'c', 'f', 'r', 'e' ],
  'f': [ 'd', 'c', 'v', 'g', 't', 'r' ],
  'g': [ 'f', 'v', 'b', 'h', 'y', 't' ],
  'h': [ 'g', 'b', 'n', 'j', 'u', 'y' ],
  'j': [ 'h', 'n', 'm', 'k', 'i', 'u' ],
  'k': [ 'j', 'm', 'l', 'o', 'i' ],
  'l': [ 'k', 'p', 'o' ],
  'z': [ 'x', 's', 'a' ],
  'x': [ 'z', 'c', 'd', 's' ],
  'c': [ 'x', 'v', 'f', 'd' ],
  'v': [ 'c', 'b', 'g', 'f' ],
  'b': [ 'v', 'n', 'h', 'g' ],
  'n': [ 'b', 'm', 'j', 'h' ],
  'm': [ 'n', 'k', 'j' ]
}


def get_keyboard_miss_typos(word):
    '''
    >>> get_keyboard_miss_typos('cat') == { \
            'xat', 'vat', 'fat', 'dat', 'czt', 'cst', 'cwt', \
            'cqt', 'car', 'caf', 'cag', 'cay', 'ca6', 'ca5' \
        }
    True
    >>> get_keyboard_miss_typos('Cat') == { \
            'Xat', 'Vat', 'Fat', 'Dat', 'Czt', 'Cst', 'Cwt', \
            'Cqt', 'Car', 'Caf', 'Cag', 'Cay', 'Ca6', 'Ca5' \
        }
    True
    '''
    typos = set()
    for i in range(len(word)):
        replacements = KEY_MISHITS_MAP.get(word[i].lower()) or []
        for replacement in replacements:
            if word[i].isupper():
                replacement = replacement.upper()
            typo = word[:i] + replacement + word[i+1:]
            typos.add(typo)
    return typos

def get_missing_letter_typos(word):
    '''
    >>> get_missing_letter_typos('cat') == {'at', 'ct', 'ca'}
    True
    '''
    typos = set()
    for i in range(len(word)):
        typo = word[:i] + word[i+1:]
        typos.add(typo)
    return typos


def get_mixed_letter_typos(word):
    '''
    >>> get_mixed_letter_typos('cat') == {'act', 'cta'}
    True
    '''
    typos = set()
    for i in range(len(word) - 1):
        typo = word[:i] + word[i+1] + word[i] + word[i+2:]
        if typo != word:
            typos.add(typo)
    return typos


def get_double_letter_typos(word):
    '''
    >>> get_double_letter_typos('cat') == {'ccat', 'caat', 'catt'}
    True
    '''
    typos = set()
    for i in range(len(word)):
        typo = word[:i] + word[i] + word[i:]
        typos.add(typo)
    return typos


def typos(word):
    '''
    >>> isinstance(typos('cat'), set)
    >>> len(typos('cat')) > 0
    '''
    sets = [get_keyboard_miss_typos(word),
            get_mixed_letter_typos(word),
            get_double_letter_typos(word),
            get_missing_letter_typos(word)]
    return set.union(*sets)

