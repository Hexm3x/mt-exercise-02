#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# individual chosen text data (a book in Gutenberg.org)
# download a different interesting data set!
#<<creature
mkdir -p $data/creature

mkdir -p $data/creature/raw

wget https://www.gutenberg.org/cache/epub/4018/pg4018.txt
mv pg4018.txt $data/creature/raw/story.txt

# preprocess slightly

cat $data/creature/raw/story.txt | python $base/scripts/preprocess_raw.py > $data/creature/raw/story.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/creature/raw/story.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/creature/raw/story.preprocessed.txt

# split into train, valid and test

head -n 440 $data/creature/raw/story.preprocessed.txt | tail -n 400 > $data/creature/valid.txt
head -n 840 $data/creature/raw/story.preprocessed.txt | tail -n 400 > $data/creature/test.txt
tail -n 3075 $data/creature/raw/story.preprocessed.txt | head -n 2955 > $data/creature/train.txt
#creature

# default
<<grimmdata
mkdir -p $data/grimm

mkdir -p $data/grimm/raw

wget https://www.gutenberg.org/files/52521/52521-0.txt
mv 52521-0.txt $data/grimm/raw/tales.txt

# preprocess slightly

cat $data/grimm/raw/tales.txt | python $base/scripts/preprocess_raw.py > $data/grimm/raw/tales.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/grimm/raw/tales.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/grimm/raw/tales.preprocessed.txt

# split into train, valid and test

head -n 440 $data/grimm/raw/tales.preprocessed.txt | tail -n 400 > $data/grimm/valid.txt
head -n 840 $data/grimm/raw/tales.preprocessed.txt | tail -n 400 > $data/grimm/test.txt
tail -n 3075 $data/grimm/raw/tales.preprocessed.txt | head -n 2955 > $data/grimm/train.txt
grimmdata

# to test the process, I used a smaller dataset - you can ignore this section
<<mocktestdata
mkdir -p $data/mockpoem

mkdir -p $data/mockpoem/raw

wget https://www.gutenberg.org/cache/epub/28218/pg28218.txt
mv pg28218.txt $data/mockpoem/raw/punctuation.txt

# preprocess slightly

cat $data/mockpoem/raw/punctuation.txt | python $base/scripts/preprocess_raw.py > $data/mockpoem/raw/punctuation.cleaned.txt
# tokenize, fix vocabulary upper bound

cat $data/mockpoem/raw/punctuation.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 500 --tokenize --lang "en" --sent-tokenize > \
    $data/mockpoem/raw/punctuation.preprocessed.txt

# split into train, valid and test

head -n 440 $data/mockpoem/raw/punctuation.preprocessed.txt | tail -n 50 > $data/mockpoem/valid.txt
head -n 840 $data/mockpoem/raw/punctuation.preprocessed.txt | tail -n 50 > $data/mockpoem/test.txt
tail -n 3075 $data/mockpoem/raw/punctuation.preprocessed.txt | head -n 150 > $data/mockpoem/train.txt
mocktestdata