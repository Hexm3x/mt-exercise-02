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

Task 2:
I do not know how to do the table as it was shown in the example. I thought it would be alright if the tables are switched (header = perplexity | Epoch), as the epochs always stay the same. With this approach, I can add all ppl in a single row. A new row is created after each run. With this, it is possible to store all different dropout from all language model in a single ppl file (there are 3 files, for each kind of perplexity).
Rows are seperated with "-" the columns are seperated with "|". For a beautiful output, the visualisation.py should be considered.

I also did not know how to put the perplexity files (which are located in an adequate folder) outside of the "world_language_model" folder (especially if the path changes?), so I copied this folder to the root folder (so that the visualisation can work) - honestly, my laptop has had problems with directories (Windows-user with wsl here), so maybe this would not have been a problem with a different system.

I do know how to put the newly trained model.pt files in a new folder (in the ./scripts/train.sh) but for some reason my laptop did not want to start the model (it did not even show the first epoch after maybe 5-10 min), so I decided to manually make a folder (Task_2_models) and put the files in there - maybe there has been an error at my part, though I considered it to be the easiest to do it manually.

The Transformer model shows a warning, though since it is only a warning for better inference performance, I ignored it (besides, it could maybe work for another system, as the problem is found in the transformer.py)
My laptop takes a lot of time to train, so I only did 4 different dropout values for the 5 language models (0.0, 0.2, 0.5, 1.0). I hope this is alright. (Training three models apparently took 30333 seconds - I did the other two seperatly / had to take breaks, so I do not know how long they took).

I have problems with the visualization.py, as the output is not interpretable for me (I think the issue is, that the .log files are sorted differently then what the plt expects and df.transpose() did not do the trick). I moved on with this problem, as for the task itself looking at the .log file (aka the tables) and the faulty linecharts are good enough for me to compare the models (although it is not the best of course).

# Changes
Task 1:
I made changes in:
main.py, generate.py (due to errors)
download_data.sh train.sh, generate.sh (working on subtasks, downloading the file that I want, training on it with different values, generating with different values - these different values are sometimes commented out, if not mandatorily needed - please check the comments for further explanation)

I created:
data (with folder creature, grimm (these all contain a "raw" folder (with the raw text, the preprocessed one and the cleaned version) and test, train and valid file), wikitext-2 (in "data/wikitext-2" and in "/pytorch-examples/word_language_model/data/wikitext-2" you can see the example data) )
models (with the file that got generated after "./scripts/train.sh" was run)
samples (with the files that got generated after "./scripts/generate.sh" was run)
alt_models (with the files that got generated after "./scripts/train.sh" was run and were not necessary - for my own, to understand the code a bit better - I made the folder manually and manually took in all files)
alt_samples (with the files that got generated after "./scripts/generate.sh" was run and were not necessary - for my own, to understand the code a bit better - I made the folder manually and manually took in all files)

Task 2:
I made changes in:
main.py, train.sh, generate.sh, install_packages.sh, generate.sh (working on subtask)

I created:
Task_2_models (with the .pt files) - I manually created that folder and manually put the files in there as the train.sh did not start (or takes too much more time to start).
main.py in the root folder (basically a copy-paste)
Log_Perplexities with the files for each kind of perplexity (in .log files - do not change the --log name from task 2, I did not check if it works with other filenames. The files in there are in a table format.)
visualization.py in the root folder (which outputs a folder "Log_Perplexities_Linecharts" with 3 .png files in it - it does not look correct as I cannot interpret it.)
Task_2_samples (with the generated text samples) - as this command generates the folder, I assume that the other folders / directory (which I made manually / replaced manually) could have been created as well, but my computer was having troubles with generating these folders as it took even longer to get the code done.

# Download (Task 2 - run ./scripts/install_packages.sh)
pandas (pip install pandas)
matplotlib (pip install -U matplotlib --prefer-binary)

# Steps get to the results of task 2 (assuming that task 1 has been run before / files from task 1 exists)
./scripts/install_packages.sh
./scripts/train.sh (assuming that main.py is in "tools/pytorch-examples/word_language_model")
Manually make a folder in "./mt-exercise-2" called "Task_2_models", put in the generated .pt files (located in "tools/pytorch-examples/word_language_model")
Manually put the folder called "Log_Perplexities" in "./mt-exercise-2" (the folder is located in "tools/pytorch-examples/word_language_model")
python3 visualization.py Log_Perplexities
./scripts/generate.sh

For further understanding and explainations, please look at the codes (especially main.py and visualization.py) and the .pdf file