#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models

num_threads=4
device=""

SECONDS=0

# Trained with the default value
#NOTE - model_epochs_40_logint_100_emsize_200_nhid_200_drop_05
#<<model_epochs_40_logint_100_emsize_200_nhid_200_drop_05
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/creature \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.5 --tied \
        --save $models/model_epochs_40_logint_100_emsize_200_nhid_200_drop_05.pt
)
<<Results_model_epochs_40_logint_100_emsize_200_nhid_200_drop_05
| epoch  40 |   100/  121 batches | lr 20.00 | ms/batch 491.17 | loss  3.70 | ppl    40.28
-----------------------------------------------------------------------------------------
| end of epoch  40 | time: 63.31s | valid loss  2.98 | valid ppl    19.66
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  3.06 | test ppl    21.37
=========================================================================================
time taken:
2604 seconds
Results_model_epochs_40_logint_100_emsize_200_nhid_200_drop_05
#model_epochs_40_logint_100_emsize_200_nhid_200_drop_05

#NOTE - training with different hyperparameter settings (epochs, log intervals, emsize, nhid)
# To enable the training with different hyperparameters, comment out the "<<[name]" and the "[name]"
# To not get confused with the submitted files, I made an alternative folder for my experimentation
<<alt_models
alt_models=$base/alt_models

mkdir -p $alt_models
alt_models

<<model_epochs_30_logint_75_emsize_150_nhid_150_drop_05
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/creature \
        --epochs 30 \
        --log-interval 75 \
        --emsize 150 --nhid 150 --dropout 0.5 --tied \
        --save $alt_models/model_epochs_30_logint_75_emsize_150_nhid_150_drop_05.pt
)
<<Results_model_epochs_30_logint_75_emsize_150_nhid_150_drop_05
| epoch  30 |    75/  121 batches | lr 20.00 | ms/batch 328.10 | loss  4.06 | ppl    58.20
-----------------------------------------------------------------------------------------
| end of epoch  30 | time: 41.69s | valid loss  3.52 | valid ppl    33.74
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  3.61 | test ppl    36.78
=========================================================================================
time taken:
2182 seconds
Results_model_epochs_30_logint_75_emsize_150_nhid_150_drop_05
model_epochs_30_logint_75_emsize_150_nhid_150_drop_05

#NOTE - model_epochs_25_logint_75_emsize_100_nhid_100_drop_05
<<model_epochs_25_logint_75_emsize_100_nhid_100_drop_05
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/creature \
        --epochs 25 \
        --log-interval 75 \
        --emsize 100 --nhid 100 --dropout 0.5 --tied \
        --save $alt_models/model_epochs_25_logint_75_emsize_100_nhid_100_drop_05.pt
)
<<Results_model_epochs_25_logint_75_emsize_100_nhid_100_drop_05
| epoch  25 |    75/  121 batches | lr 5.00 | ms/batch 368.78 | loss  4.27 | ppl    71.69
-----------------------------------------------------------------------------------------
| end of epoch  25 | time: 49.13s | valid loss  3.89 | valid ppl    48.71
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  3.95 | test ppl    51.89
=========================================================================================
time taken:
1172 seconds
Results_model_epochs_25_logint_75_emsize_100_nhid_100_drop_05
model_epochs_25_logint_75_emsize_100_nhid_100_drop_05

#NOTE - model_epochs_20_logint_50_emsize_100_nhid_100_drop_05
<<model_epochs_20_logint_50_emsize_100_nhid_100_drop_05
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/creature \
        --epochs 20 \
        --log-interval 50 \
        --emsize 100 --nhid 100 --dropout 0.5 --tied \
        --save $alt_models/model_epochs_20_logint_50_emsize_100_nhid_100_drop_05.pt
)
<<Results_model_epochs_20_logint_50_emsize_100_nhid_100_drop_05
| epoch  20 |    50/  121 batches | lr 20.00 | ms/batch 311.78 | loss  4.48 | ppl    87.80
| epoch  20 |   100/  121 batches | lr 20.00 | ms/batch 262.95 | loss  4.39 | ppl    80.87
-----------------------------------------------------------------------------------------
| end of epoch  20 | time: 36.02s | valid loss  4.04 | valid ppl    56.78
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  4.10 | test ppl    60.13
=========================================================================================
time taken:
855 seconds
Results_model_epochs_20_logint_50_emsize_100_nhid_100_drop_05
model_epochs_20_logint_50_emsize_100_nhid_100_drop_05

#NOTE - model_epochs_5_logint_61_emsize_20_nhid_20_drop_05
<<model_epochs_5_logint_61_emsize_20_nhid_20_drop_05
(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/creature \
        --epochs 5 \
        --log-interval 61 \
        --emsize 20 --nhid 20 --dropout 0.5 --tied \
        --save $alt_models/model_epochs_5_logint_61_emsize_20_nhid_20_drop_05.pt
)
<<Results_model_epochs_5_logint_61_emsize_20_nhid_20_drop_05
| epoch   5 |    61/  121 batches | lr 20.00 | ms/batch 135.62 | loss  5.65 | ppl   284.18
-----------------------------------------------------------------------------------------
| end of epoch   5 | time: 17.98s | valid loss  5.35 | valid ppl   210.52
-----------------------------------------------------------------------------------------
=========================================================================================
| End of training | test loss  5.34 | test ppl   207.52
=========================================================================================
time taken:
182 seconds
Results_model_epochs_5_logint_61_emsize_20_nhid_20_drop_05
model_epochs_5_logint_61_emsize_20_nhid_20_drop_05

# NOTE - echo
echo "time taken:"
echo "$SECONDS seconds"