echo "--- Processing harmov ---"
python analysis.py segmentation \
    ../exps/segmentation/small-audio/harmov \
    --known_patterns ../data/known-patterns/known_sequences_ts_split.pkl \
    --resampling_size 30 --metric_name dtw --n_workers 6

echo "--- Processing FLUSS ---"
python analysis.py segmentation \
    ../exps/segmentation/small-audio/fluss_segmentation \
    --known_patterns ../data/known-patterns/known_sequences_ts_split.pkl \
    --resampling_size 30 --metric_name dtw --n_workers 6

echo "--- Processing UniformSplit ---"
python analysis.py segmentation \
    ../exps/segmentation/small-audio/uniform_split \
    --known_patterns ../data/known-patterns/known_sequences_ts_split.pkl \
    --resampling_size 30 --metric_name dtw --n_workers 6

# echo "--- Processing RandomSplit ---"
# python analysis.py segmentation \
#     ../exps/segmentation/small-audio/random_split \
#     --known_patterns ../data/known-patterns/known_sequences_ts_split.pkl \
#     --resampling_size 30 --metric_name dtw --n_workers 6

python analysis.py segwrite ../exps/segmentation/small-audio/