#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4
device=""

# To enable the individual generation, comment out the "<<[name]" and the "[name]"
# this ensures that the model will take a shorter time span (not generating different files at once)
# Generated with the default value

#<<model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_1
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 100 \
        --checkpoint $models/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05.pt \
        --outf $samples/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_1.txt
)
#model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_1

#<<model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_075
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 100 \
        --temperature 0.75 \
        --checkpoint $models/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05.pt \
        --outf $samples/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_075.txt
)
#model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_075

#<<model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_075
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 200 \
        --temperature 0.75 \
        --checkpoint $models/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05.pt \
        --outf $samples/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_075.txt
)
#model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_075

#<<model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_1
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 200 \
        --checkpoint $models/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05.pt \
        --outf $samples/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_1.txt
)
#model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_1

#<<model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_05
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 100 \
        --temperature 0.5 \
        --checkpoint $models/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05.pt \
        --outf $samples/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_05.txt
)
#model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_05

#<<model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_05
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 200 \
        --temperature 0.5 \
        --checkpoint $models/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05.pt \
        --outf $samples/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_05.txt
)
#model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_05

#<<model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_025
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 100 \
        --temperature 0.25 \
        --checkpoint $models/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05.pt \
        --outf $samples/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_025.txt
)
#model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_100_temperature_025

#<<model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_025
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 200 \
        --temperature 0.25 \
        --checkpoint $models/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05.pt \
        --outf $samples/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_025.txt
)
#model_epochs_40_logint_100_emsize_200_nhid_200_drop_05_sample_words_200_temperature_05

#NOTE - generating with different hyperparameter settings (epochs, log intervals, emsize, nhid)
# To enable the generation with different hyperparameters, comment out the "<<[name]" and the "[name]"
# To not get confused with the submitted files, I made an alternative folder for my experimentation
<<model_epochs_30_logint_75_emsize_150_nhid_150_drop_05_sample_words_100_temperature_1
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 100 \
        --checkpoint $models/model_epochs_30_logint_75_emsize_150_nhid_150_drop_05.pt \
        --outf $samples/model_epochs_30_logint_75_emsize_150_nhid_150_drop_05_sample_words_100_temperature_1.txt
)
model_epochs_30_logint_75_emsize_150_nhid_150_drop_05_sample_words_100_temperature_1

<<model_epochs_30_logint_75_emsize_150_nhid_150_drop_05_sample_words_100_temperature_1
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 100 \
        --checkpoint $models/model_epochs_30_logint_75_emsize_150_nhid_150_drop_05.pt \
        --outf $samples/model_epochs_30_logint_75_emsize_150_nhid_150_drop_05_sample_words_100_temperature_1.txt
)
model_epochs_30_logint_75_emsize_150_nhid_150_drop_05_sample_words_100_temperature_1

<<model_epochs_25_logint_75_emsize_100_nhid_100_drop_05_sample_words_100_temperature_1
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 100 \
        --checkpoint $models/model_epochs_25_logint_75_emsize_100_nhid_100_drop_05.pt \
        --outf $samples/model_epochs_25_logint_75_emsize_100_nhid_100_drop_05_sample_words_100_temperature_1.txt
)
model_epochs_25_logint_75_emsize_100_nhid_100_drop_05_sample_words_100_temperature_1

<<model_epochs_20_logint_50_emsize_100_nhid_100_drop_05_sample_words_100_temperature_1
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 100 \
        --checkpoint $models/model_epochs_20_logint_50_emsize_100_nhid_100_drop_05.pt \
        --outf $samples/model_epochs_20_logint_50_emsize_100_nhid_100_drop_05_sample_words_100_temperature_1.txt
)
model_epochs_20_logint_50_emsize_100_nhid_100_drop_05_sample_words_100_temperature_1

<<model_epochs_5_logint_61_emsize_20_nhid_20_drop_05_sample_words_100_temperature_1
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/creature \
        --words 100 \
        --checkpoint $models/model_epochs_5_logint_61_emsize_20_nhid_20_drop_05.pt \
        --outf $samples/model_epochs_5_logint_61_emsize_20_nhid_20_drop_05_sample_words_100_temperature_1.txt
)
model_epochs_5_logint_61_emsize_20_nhid_20_drop_05_sample_words_100_temperature_1
