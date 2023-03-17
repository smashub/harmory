"""
Script to parse the dataset and process the known patterns recursively.
"""
import json
from pathlib import Path


def clean_blocks(string):
    if '(' in string and ')' in string:
        brackets = string[string.find('(') + 1:string.find(')')]
        no_space = brackets.replace(' ', '')
        return string.replace(brackets, no_space)
    return string


def store_blocks(blocks_path: str) -> None:
    """
    Store the blocks in a json file
    :param blocks_path: the path to the blocks.txt file
    :type blocks_path: str
    :return: None
    :rtype: None
    """
    with open(blocks_path, 'r') as f:
        blocks = f.read()

    blocks = [[z.strip().lstrip('(').rstrip(')') for z in x.split('\n')] for x
              in
              blocks.split('\n\n') if not x.startswith('//')]
    block_dict = {}
    for block in blocks:
        if block[0].startswith('defbrick'):
            block_meta = block[0].split(' ')[1:]
            block_name = ''.join([x for x in block_meta[:-3]])
            block_meta = block_meta[-3:]
            components = [clean_blocks(x).split(' ') for x in block[1:]]
            components = [
                {'type': x[0], 'value': x[1], 'duration': x[2]} if len(
                    x) == 3 else {'type': x[0], 'name': x[1], 'value': x[2],
                                  'duration': x[3]} for x in components]
            mode, typology, key = block_meta
            block_dict[block_name] = {'mode': mode, 'block_type': typology,
                                      'block_key': key,
                                      'components': components}

    json.dump(block_dict, open('blocks.json', 'w'), indent=4)


if __name__ == '__main__':
    store_blocks('blocks.txt')
