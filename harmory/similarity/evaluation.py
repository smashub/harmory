"""
Utility functions for evaluating similarity measures.
The functions in this module are the same as those in the TPSD paper.
"""
import json
from typing import Any

import joblib
import pandas as pd
from jellyfish import jaro_similarity


def get_covers(dataset: list, string_match_threshold: float = .98) -> dict:
    """
    Get the cover songs from the dataset
    Parameters
    ----------
    dataset : list
        List of tuples containing the dataset
    string_match_threshold : float
        Threshold for the string similarity measure
    Returns
    -------
    list[tuple]
        List of tuples containing the cover songs
    """
    covers = {}
    titles = [x[0] for x in dataset]
    for track in titles:
        cover = [t for t in titles if
                 jaro_similarity(track.split('$')[-1],
                                 t.split('$')[-1]) >= string_match_threshold]
        if len(cover) > 1 and all(
                [jaro_similarity(track.split('$')[-1],
                                 t.split('$')[-1]) < string_match_threshold for
                 t in
                 covers.keys()]):
            covers[track] = cover
    covers = {k: v for k, v in
              sorted(covers.items(), key=lambda item: item[1], reverse=True)}
    print(len(covers))
    json.dump(covers, open('../../exps/results/covers.json', 'w'), indent=4)
    return covers


def covers_ranking(results: list[tuple]) -> dict[Any, Any]:
    """
    Evaluate the results of a similarity measure using standard evaluation
    metrics
    Parameters
    ----------
    results : list[tuple]
        List of tuples containing the results of the similarity measure
    Returns
    -------
    dict[Any, Any]
        Dictionary containing the ranking of the cover songs
    """
    weighted_results = []
    for res in results:
        if jaro_similarity(res[0].split('$')[-1], res[1].split('$')[-1]) >= .98:
            if res[0] != res[1]:
                weighted_results.append([*res, True])
        else:
            weighted_results.append([*res, False])

    df = pd.DataFrame(weighted_results,
                      columns=['title1', 'title2', 'tpsd_distance',
                               'distance_alert'])
    ranking = {}
    for track in df['title1'].unique():
        if df[(df['title1'] == track) & (df['distance_alert'] == True)].shape[
            0] != 0:
            df_track = df[df['title1'] == track]
            df_track = df_track.sort_values(by=['tpsd_distance'],
                                            ascending=True)
            ranking[track] = df_track['title2'].tolist()
    return ranking


def evaluate(covers_dict: dict, covers_ranking: dict) -> tuple[float, float]:
    """
    Evaluates the covers ranking
    :param covers_dict: the dictionary with the covers' titles
    :type covers_dict: dict[str, list[str]]
    :param covers_ranking: the dictionary with the covers ranking
    :type covers_ranking: dict[str, list[str]]
    :return: None
    """
    first_tier, second_tier = {}, {}
    for track in covers_dict.keys():
        class_size = len(covers_dict[track]) - 1
        class_size_second = 2 * len(covers_dict[track]) - 1
        if track in covers_ranking.keys():
            first_tier[track] = len(
                [x for x in covers_ranking[track][:class_size] if x in
                 list(covers_dict[track])]) / class_size
            second_tier[track] = len(
                [x for x in covers_ranking[track][:class_size_second] if x in
                 list(covers_dict[track])]) / class_size

    return sum(first_tier.values()) / len(first_tier), sum(
        second_tier.values()) / len(second_tier)


if __name__ == '__main__':
    data = joblib.load(
        '../../exps/datasets/cover-song-data-jams-timeseries/timeseries_offset.pkl')
    covers = get_covers(data)
    covers_ranking(data)
