# Press the green button in the gutter to run the script.
import os
import time

import pandas as pd

source_filename = 'energy_performance'
file_extension = '.csv'

cache_folder = 'cache'

########################################################################################################################
# Task 1:  Input Data
########################################################################################################################

def task1(df):
    OutputFormat.print_header('h1', 'Task 1: Input Data')

    print('Separating labels and targets...')
    labels, targets = separate_labels_and_targets(df)
    print('Labels:')
    for label in labels.columns:
        print(f'  - {label}')

    print('\nTargets:')
    for target in targets.columns:
        print(f'  - {target}')


    OutputFormat.print_divider('h2')


    # determine max and min values for each target
    max_target = targets.max()
    min_target = targets.min()


    # create a DataFrame for max and min values
    max_min_df = pd.DataFrame({'Max': max_target, 'Min': min_target})

    # print as table
    print('Max and Min values for each target:')
    print(max_min_df)


def separate_labels_and_targets(df):
    # Separate the labels and targets
    labels = df.iloc[:, 0:8]
    targets = df.iloc[:, 8:10]

    return labels, targets


########################################################################################################################
# Task 2: Model Function
########################################################################################################################

def task2(df):
    OutputFormat.print_header('h1', 'Task 2: Model Function')
    pass

########################################################################################################################
# Task 3: Linearization
########################################################################################################################

def task3(df):
    OutputFormat.print_header('h1', 'Task 3: Linearization')
    pass

########################################################################################################################
# Task 4: Parameter Update
########################################################################################################################

def task4(df):
    OutputFormat.print_header('h1', 'Task 4: Parameter Update')
    pass

########################################################################################################################
# Task 5: Regression
########################################################################################################################

def task5(df):
    OutputFormat.print_header('h1', 'Task 5: Regression')
    pass

########################################################################################################################
# Task 6: Model Selection
########################################################################################################################

def task6(df):
    OutputFormat.print_header('h1', 'Task 6: Model Selection')
    pass

########################################################################################################################
# Task 7:  Evaluation and Visualisation of Results
########################################################################################################################

def task7(df):
    OutputFormat.print_header('h1', 'Task 7:  Evaluation and Visualisation of Results')
    pass

########################################################################################################################
# Formatting
########################################################################################################################

class OutputFormat:
    SECTION_DIVIDER = {
        'title': '█',
        'h1': '#',
        'h2': '*',
        'h3': '-'
    }

    def __init__(self):
        pass

    @staticmethod
    def print_header(section, text):
        print('\n' + OutputFormat.SECTION_DIVIDER[section] * 80)
        print(text.center(80, ' '))
        print(str(OutputFormat.SECTION_DIVIDER[section] * 80) + '\n')

    @staticmethod
    def print_divider(section):
        print("\n" + OutputFormat.SECTION_DIVIDER[section] * 80 + "\n")

    @staticmethod
    # Function to display a progress bar so the user knows the program is still running and how far along it is
    def progressbar(i, upper_range, start_time):
        # Calculate the percentage of completion
        percentage = (i / (upper_range - 1)) * 100
        # Calculate the number of '█' characters to display
        num_blocks = int(percentage/2)
        # Calculate elapsed time and estimated remaining time

        elapsed_time = time.time() - start_time
        if percentage > 0:
            estimated_total_time = elapsed_time / (percentage / 100)
            remaining_time = estimated_total_time - elapsed_time
        else:
            remaining_time = 0

        # Create the progress bar string
        progress_string = f'\r{("█" * num_blocks)}{("_" * (50 - num_blocks))} {percentage:.2f}% | Elapsed: {elapsed_time:.2f}s | Remaining: {remaining_time:.2f}s'
        if i == upper_range - 1:
            print(progress_string)
        else:
            print(progress_string, end='', flush=True)

########################################################################################################################
# Main Function
########################################################################################################################

def main():
    OutputFormat.print_header('title', 'Machine Learning Assignment 3')

    start_time = time.time()

    # Create a cache folder if it does not exist
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    original_data = read_data()

    task1(original_data.copy())
    task2(original_data.copy())
    task3(original_data.copy())
    task4(original_data.copy())
    task5(original_data.copy())
    task6(original_data.copy())
    task7(original_data.copy())

    print(f'\nTotal Runtime: {time.time() - start_time:.2f}s')

def read_data():
    df = pd.read_csv(source_filename + file_extension, header=0)
    return df


if __name__ == '__main__':
    main()