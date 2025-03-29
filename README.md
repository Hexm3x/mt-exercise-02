# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/marpng/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

# Problems and Solutions
Task 1:
While using the scripts, I encountered an unpickling error while loading the "./scripts.train" as it did not show the end result of the training (in the train script) even though it went through all epochs. 
To solve this problem, I had to change line 254 in the main.py ("model = torch.load(f)" to "model = torch.load(f, weights_only=False)") and I had to change line 56 in generate.py ("model = torch.load(f, map_location=device)" to "model = torch.load(f, map_location=device, weights_only=False)")

# Changes
Task 1:
I made changes in:
main.py, generate.py (due to errors)
download_data.sh train.sh, generate.sh (working on subtasks, downloading the file that I want, training on it with different values, generating with different values - these different values are sometimes commented out, if not mandatorily needed - please check the comments for further explanation)

I created:
data (with folder creature, grimm (these all contain a "raw" folder (with the raw text, the preprocessed one and the cleaned version) and test, train and valid file), wikitext-2 (in "data/wikitext-2" and in "/pytorch-examples/word_language_model/data/wikitext-2" you can see the example data) )
models (with the file that got generated after "./scripts/train.sh" was run)
samples (with the files that got generated after "./scripts/generate.sh" was run)
alt_models (with the files that got generated after "./scripts/train.sh" was run - for my own, to understand the code a bit better)
alt_samples (with the files that got generated after "./scripts/generate.sh" was run - for my own, to understand the code a bit better)