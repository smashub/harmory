"""
Script to parse the dataset and process the known patterns recursively.
"""
import json
import re
from collections import defaultdict
from pathlib import Path


def clean_label(block_label: str) -> list[str]:
    """
    Clean the label of the block
    :param block_label: the label of the block
    :type block_label: str
    :return: the cleaned label
    :rtype: str
    """
    cleaned_lebel = re.split(r'\((.*?)\)', block_label)
    return list(filter(None, cleaned_lebel))


def clean_blocks(string: str) -> str:
    """
    Utility function to clean the blocks string parsed from the blocks.txt file
    :param string: the string to clean up
    :type string: str
    :return: the cleaned string
    :rtype: str
    """
    if '(' in string and ')' in string:
        brackets = string[string.find('(') + 1:string.find(')')]
        no_space = brackets.replace(' ', '')
        return string.replace(brackets, no_space)
    return string


def store_blocks(blocks_path: str | Path,
                 output_path: str | Path = './blocks.json') -> dict:
    """
    Store the blocks in a json file
    :param blocks_path: the path to the blocks.txt file
    :type blocks_path: str | Path
    :param output_path: the path to the output file
    :type output_path: str | Path
    :default output_path: blocks.json
    :return: a dictionary with the blocks and their components
    :rtype: dict
    """
    if isinstance(blocks_path, str):
        blocks_path = Path(blocks_path)
    if isinstance(output_path, str):
        output_path = Path(output_path)

    with open(blocks_path, 'r') as f:
        blocks = f.read()

    blocks = [[z.strip().lstrip('(').rstrip(')') for z in x.split('\n')] for x
              in
              blocks.split('\n\n') if not x.startswith('//')]
    block_dict = defaultdict(list)
    for block in blocks:
        if block[0].startswith('defbrick'):
            block_meta = block[0].split(' ')[1:]
            block_name = ''.join([x for x in block_meta[:-3]])
            block_name = clean_label(block_name)
            block_name, block_variant = block_name if len(
                block_name) == 2 else (block_name[0], None)
            block_meta = block_meta[-3:]
            components = [clean_blocks(x).split(' ') for x in block[1:]]
            components = [
                {'type': x[0], 'value': x[1], 'duration': x[2]} if len(
                    x) == 3 else {'type': x[0], 'name': x[1], 'value': x[2],
                                  'duration': x[3]} for x in components]
            mode, typology, key = block_meta

            block_dict[block_name].append(
                {'variant_name': block_variant,
                 'mode': mode,
                 'block_type': typology,
                 'block_key': key,
                 'components': components})

    json.dump(block_dict, open(output_path, 'w'), indent=4)

    return block_dict


if __name__ == '__main__':
    store_blocks('blocks.txt')
