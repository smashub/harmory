"""

"""

import json
from pathlib import Path


def clean_blocks(string):
    if '(' in string and ')' in string:
        brackets = string[string.find('(') + 1:string.find(')')]
        no_space = brackets.replace(' ', '')
        return string.replace(brackets, no_space)
    return string


if __name__ == '__main__':
    pass
