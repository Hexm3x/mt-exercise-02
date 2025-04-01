import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import re
import numpy as np

"""
Tool to visualize the results that are stored in the Log_Perplexities folder.
"""
if len(sys.argv) < 2:
    print('Usage: visualization.py [foldername]')
    sys.exit(1)
perplexity_files = sys.argv[1]


def get_the_log_files_from_the_folder():
    try:
        files_with_perplexities = os.listdir(perplexity_files)
        if files_with_perplexities and len(files_with_perplexities) == 3:
            # ChatGPT is a great aid to shorten 5-line-for-loops into these one-liners
            log_files = [
                files for files in files_with_perplexities if files.endswith(".log")]
            if len(log_files) == 3:
                return (sorted(log_files))

        else:
            print("Your folder is either empty or does not contain 3 .log files.")
            sys.exit(1)

    except FileNotFoundError:
        print("This folder does not exist.")
        sys.exit(1)


def path_stuff(I_need_an_iteration):
    for log_perplexities in I_need_an_iteration:
        Log_Perplexities_joined = os.path.join(
            perplexity_files, log_perplexities)
        # Honestly, I asked ChatGPT as I got to the point where I iterated over all files instead of each file individually (I had the stuff that is now in extract_data_from_file() in path_stuff())
        data = extract_data_from_file(Log_Perplexities_joined)
        visualization(data, log_perplexities)


def extract_data_from_file(log_perplexities_path):
    data = []
    with open(log_perplexities_path, 'r', encoding="UTF-8") as extracted_ppl:
        extracted_ppl_read = extracted_ppl.read().strip()
        extracted_ppl_list = re.sub(
            r"(\n-+)", "", extracted_ppl_read).split("\n")
        for entry in extracted_ppl_list:
            line_entry = re.sub(
                r"( \||\| | \| |\|)", " ", entry)
            data.append([line_entry])
        return data


def get_model_name(line_entry):
    pattern = r"(\w+ Dropout \d\.\d)"
    model_name_splitter = re.split(pattern, line_entry)
    model_name = model_name_splitter[0].strip()
    # yet again, ChatGPT shortening the long for-loops
    perplexities = [p.strip() for p in model_name_splitter[1:] if p.strip()]

    return model_name, perplexities

# Had a lot of help of ChatGPT since I had to correctly split everything
# Nevertheless, I could not figure out how to do the correct labeling (as the file should be transposed but this did not really work - I do not have the time or energy to solve this)


def visualization(data, log_perp_file):
    picturename = f"{perplexity_files}_Linecharts"
    if not os.path.isdir(picturename):
        os.makedirs(picturename)
    columns = data[0][0]
    columns = re.split(
        r'(Perplexity|Valid. perplexity|Epoch \d+|Model Name with Dropout|Test perplexity|)', columns.strip())
    columns = [column.strip() for column in columns if column.strip()]
    rows = data[1:]
    clean_rows = []
    for row in rows:
        row_data = re.split(
            r"(\w+ Dropout \d\.\d|\d+\.\d+|nan)", row[0].strip())
        row_data = [r.strip() for r in row_data if r.strip()]
        if len(row_data) < len(columns):
            row_data.extend([np.nan] * (len(columns) - len(row_data)))
        clean_rows.append(row_data)

    df = pd.DataFrame(clean_rows, columns=columns)
    df = df.fillna(0)

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].astype(float)

    plt.figure(figsize=(10, 6))
    for column in columns[1:]:
        plt.plot(df[column], marker='o', label=column)

    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title(f"Perplexity Comparison - {log_perp_file}")
    plt.legend(title="Models", loc="best")
    plt.tight_layout()
    plt.savefig(f"./{picturename}/{log_perp_file}.png")
    plt.clf()


# get_the_log_files_from_the_folder()
log_file_extraction_help = get_the_log_files_from_the_folder()
path_stuff(log_file_extraction_help)
