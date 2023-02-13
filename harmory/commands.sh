# SANITY CHECKS on selected chord data
python create.py encode ../../choco/choco-jams/jams \
    --selection ../data/samples/mixed_audio_med.txt \
    --out_dir ../exps/segmentation/medium-audio/ \
    --n_workers 6 --debug

# ############################################################################ #
# SEGMENTATION
# ############################################################################ #

# ... on a small sample of Billboard
python create.py segment ../../choco/choco-jams/jams \
    --selection ../data/samples/billboard_small.txt \
    --out_dir ../data/structures --n_workers 4
# ... on a small sample of all audio datasets in ChoCo
python create.py segment ../../choco/choco-jams/jams \
    --selection ../data/samples/mixed_audio_small.txt \
    --out_dir ../exps/segmentation/small-audio/harmov/k%8__p%msaf__pml%24__psig%2 \
    --n_workers 6 --debug

python baselines.py ../../choco/choco-jams/jams \
    --baselines fluss_segmentation \
    --selection ../data/samples/mixed_audio_small.txt \
    --out_dir ../exps/segmentation/small-audio --n_workers 6

python baselines.py ../../choco/choco-jams/jams \
    --baselines uniform_split \
    --selection ../data/samples/mixed_audio_small.txt \
    --out_dir ../exps/segmentation/small-audio --n_workers 6

# on something bigger
python create.py segment ../../choco/choco-jams/jams \
    --selection ../data/samples/mixed_audio_med.txt \
    --out_dir ../exps/segmentation/medium-audio/harmov/k%8__p%msaf__pml%24__psig%2 \
    --n_workers 6 --debug

python baselines.py ../../choco/choco-jams/jams \
    --baselines fluss_segmentation \
    --selection ../data/samples/mixed_audio_med.txt \
    --out_dir ../exps/segmentation/medium-audio --n_workers 6

python baselines.py ../../choco/choco-jams/jams \
    --baselines uniform_split \
    --selection ../data/samples/mixed_audio_med.txt \
    --out_dir ../exps/segmentation/medium-audio --n_workers 6

# ############################################################################ #
# VALIDATION of SEGMENTATION
# ############################################################################ #

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

echo "--- Processing RandomSplit ---"
python analysis.py segmentation \
    ../exps/segmentation/small-audio/random_split \
    --known_patterns ../data/known-patterns/known_sequences_ts_split.pkl \
    --resampling_size 30 --metric_name dtw --n_workers 6

echo "--- Processing UniformSplit ---"
python analysis.py segmentation \
    ../exps/segmentation/small-audio/uniform_split \
    --known_patterns ../data/known-patterns/known_sequences_ts_split.pkl \
    --resampling_size 30 --metric_name dtw --n_workers 6

echo "--- Processing QuasiRandom ---"
python analysis.py segmentation \
    ../exps/segmentation/small-audio/quasirandom_split \
    --known_patterns ../data/known-patterns/known_sequences_ts_split.pkl \
    --resampling_size 30 --metric_name dtw --n_workers 6

python analysis.py segwrite ../exps/segmentation/small-audio/

# ############################################################################ #
# SIMILARITIES
# ############################################################################ #

python create.py similarities ../data/structures/v1 \
    --out_dir ../data/similarities/v1 --n_workers 6