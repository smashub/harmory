"""

"""

import json
from pathlib import Path


def read_blocks(blocks_path: str | Path) -> dict:
    """
    Read the blocks from the blocks.json file
    :param blocks_path: the path to the blocks.txt file
    :type blocks_path: str | Path
    :return: a dictionary with the blocks and their components
    :rtype: dict
    """
    if isinstance(blocks_path, str):
        blocks_path = Path(blocks_path)

    assert blocks_path.exists(), f'{blocks_path} does not exist'
    assert blocks_path.is_file(), f'{blocks_path} is not a file'

    return json.load(open(blocks_path, 'r'))


def get_block_components(all_blocks: dict, block_components: list) -> list:
    """
    Get the components of the blocks
    :param all_blocks: a dictionary with the blocks and their components
    :type all_blocks: dict
    :param block_components: a dictionary with the blocks and their components
    :type block_components: list
    :return: a dictionary with the blocks and their components
    :rtype: dict
    """
    results = []
    for component in block_components:
        if component['type'] == 'brick':
            component_name = component['name']
            component_value = component['value']
            component_duration = component['duration']
            retrieved_block = all_blocks[component_name]
            brick_results = []
            for retrieved_variant in retrieved_block:
                #print('ret_variant', retrieved_variant)
                # if component_value == retrieved_variant['block_key']:
                retrieved_components = retrieved_variant['components']
                brick_results.append(get_block_components(all_blocks, retrieved_components))
            results.append(brick_results if len(brick_results) > 1 else brick_results[0])
        elif component['type'] == 'chord':
            results.append(component)

    return results


def unwrap_results(results: list) -> list:
    """
    Unwrap the results from the get_block_components function
    :param results: the results from the get_block_components function
    :type results: list
    :param base: the base list to append the results to
    :type base: list | None
    :return: a list of chord sequences
    :rtype: list

    """
    unwrapped_results = []

    print('results', results)
    for result in results:
        if len(result) == 1:
            result = result[0]
        if isinstance(result, dict):
            unwrapped_results.append(result)
            print(result)
        else:
            temp = unwrapped_results.copy()
            unwrap_results(result)
    return unwrapped_results
    #print('unwrapped', unwrapped_results)



def process_blocks(blocks_path: str | Path) -> list:
    """
    Process the blocks from the blocks.txt file and return a list of chord
    sequences
    :param blocks_path: the path to the blocks.json file
    :type blocks_path: str | Path
    :return: a list of chord sequences
    :rtype: list
    """
    blocks = read_blocks(blocks_path)

    for block in blocks:
        for variant in blocks[block]:
            mode = variant['mode']
            key = variant['block_key']
            components = variant['components']
            print(components)
            abc = get_block_components(blocks, components)



if __name__ == '__main__':
    # process_blocks('./blocks.json')
    abc = get_block_components(read_blocks('./blocks.json'), [
                {
                    "type": "chord",
                    "value": "Dm7",
                    "duration": "1"
                },
                {
                    "type": "brick",
                    "name": "GenDom",
                    "value": "D",
                    "duration": "1"
                },
                {
                    "type": "brick",
                    "name": "An-Approach",
                    "value": "C",
                    "duration": "*"
                }
            ])
    print(abc)
    cde = unwrap_results([{'type': 'chord', 'value': 'Dm7', 'duration': '1'}, [[{'type': 'chord', 'value': 'G7', 'duration': '1'}], [[[{'type': 'chord', 'value': 'Dm7', 'duration': '1'}], [{'type': 'chord', 'value': 'Dm7b5', 'duration': '1'}]], {'type': 'chord', 'value': 'G7', 'duration': '1'}]]])
    print('$$$', cde)
