# Assignment 3: Unsupervised Learning
# Module: Machine Learning COMP8043
# Author: Emmett Fitzharris
# Student Number: R00222357
# Date: 06/12/2024

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

source_filename = 'energy_performance'
file_extension = '.csv'

########################################################################################################################
# Task 1:  Input Data
########################################################################################################################

def task1(df):
    OutputFormat.print_header('h1', 'Task 1: Input Data')

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
    print('Max and Min values for each target:\n')
    print(max_min_df)

    return labels, targets


def separate_labels_and_targets(df):
    # Separate the labels and targets
    labels = df.iloc[:, 0:8]
    targets = df.iloc[:, 8:10]

    return labels, targets


########################################################################################################################
# Task 2: Model Function
########################################################################################################################

def polynomial_model(degree, feature_vectors, coefficients):
    result = np.zeros(feature_vectors.shape[0])
    k = 0
    for i in range(degree + 1):
        for j in range(degree + 1):
            if i + j <= degree:
                result += coefficients[k] * (feature_vectors[:, 0] ** i) * (feature_vectors[:, 1] ** j)
                k += 1
    return result

def num_parameters(degree):
    t = 0
    for i in range(degree + 1):
        for j in range(degree + 1):
            if i + j <= degree:
                t += 1
    return t

########################################################################################################################
# Task 3: Linearization
########################################################################################################################

def calculate_model_and_jacobian(degree, feature_vectors, coefficients):

    # This call calculates the estimated target vector f0 using the initial coefficients at the linearization point.
    f0 = polynomial_model(degree, feature_vectors, coefficients)

    num_samples = feature_vectors.shape[0]
    num_params = len(coefficients)

    Jacobian = np.zeros((num_samples, num_params))
    epsilon = 1e-6


    for i in range(num_params):
        coef = coefficients.copy()
        coef[i] += epsilon
        # This call calculates the model function value fi with slightly modified coefficients to compute the partial
        # derivatives for the Jacobian. The partial derivative is determined to be the difference between the model
        # function value fi and the model function value f0 divided by epsilon.
        fi = polynomial_model(degree, feature_vectors, coef)
        Jacobian[:, i] = (fi - f0) / epsilon

    return f0, Jacobian

########################################################################################################################
# Task 4: Parameter Update
########################################################################################################################

def calculate_optimal_parameter_update(training_target, estimated_target, jacobian):
    regularization_lambda = 1e-6

    # Normal matrix is calculated as the product of the Jacobian transposed and the Jacobian, plus a regularization term.
    # The "@" operator is used to perform matrix multiplication.
    # np.eye creates an identity matrix with the same number of columns as the Jacobian matrix.
    normal_matrix = jacobian.T @ jacobian + regularization_lambda * np.eye(jacobian.shape[1])

    # The residual is the difference between the training_target vector and the estimated_target vector.
    # This represents the error between the actual target values and the values predicted by the model.
    residual = training_target - estimated_target

    rhs = jacobian.T @ residual

    optimal_update = np.linalg.solve(normal_matrix, rhs)

    return optimal_update

########################################################################################################################
# Task 5: Regression
########################################################################################################################

def fit_polynomial_model(degree, training_features, training_targets, max_iterations=100, tolerance=1e-6):
    num_params = num_parameters(degree)
    coefficients = np.zeros(num_params) # parameter vector

    for iteration in range(max_iterations):
        estimated_target, jacobian = calculate_model_and_jacobian(degree, training_features, coefficients)

        optimal_update = calculate_optimal_parameter_update(training_targets, estimated_target, jacobian)

        coefficients += optimal_update # update the parameter vector

        # Initially these parameter updates are likely to be large, as the model starts from all zeros which are
        # unlikely to be the optimal values. As the model converges, the parameter updates will become smaller. The
        # residuals will also decrease as the model converges and the model becomes better fit to the data.
        #
        # As such we can use the norm of the optimal update vector as a measure of convergence and determine the optimal
        # number of iterations. Convergence is determined by the norm of the optimal update vector being less than the
        # tolerance value. For the tolerance value a small value of 1e-6 is used. This is arbitrary and can be adjusted
        # as needed. It could be tested as a parameter itself to determine the optimal value.
        if np.linalg.norm(optimal_update) < tolerance:
            # print(f'Converged in {iteration + 1} iterations.') # Uncomment to print the number of iterations to convergence
            iterations_to_convergence = iteration + 1
            break

    return coefficients

########################################################################################################################
# Task 6: Model Selection
########################################################################################################################

def task6(labels, targets):
    OutputFormat .print_header('h1', 'Task 6: Model Selection')
    features = labels.values
    heating_targets = targets.iloc[:, 0].values
    cooling_targets = targets.iloc[:, 1].values

    optimal_heating_degree, optimal_cooling_degree = evaluate_polynomial_degrees(features, heating_targets, cooling_targets)
    return optimal_heating_degree, optimal_cooling_degree


def cross_validate_polynomial_model(degree, features, targets, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    absolute_differences = []

    for train_index, test_index in kf.split(features):
        # Split the data into training and testing sets
        train_features, test_features = features[train_index], features[test_index]
        train_targets, test_targets = targets[train_index], targets[test_index]

        # Fit the polynomial model to the training data
        coefficients = fit_polynomial_model(degree, train_features, train_targets)

        # Predict the targets for the test data
        predicted_targets = polynomial_model(degree, test_features, coefficients)

        # Calculate the mean absolute difference between the predicted and actual targets
        absolute_differences.append(mean_absolute_error(test_targets, predicted_targets))

    # Return the mean of the mean absolute differences
    return np.mean(absolute_differences)

def evaluate_polynomial_degrees(features, heating_targets, cooling_targets, max_degree=2):
    heating_errors = []
    cooling_errors = []

    for degree in range(max_degree + 1):
        heating_error = cross_validate_polynomial_model(degree, features, heating_targets)
        cooling_error = cross_validate_polynomial_model(degree, features, cooling_targets)

        heating_errors.append(heating_error)
        cooling_errors.append(cooling_error)

        # Output the mean absolute difference
        print(f'Degree {degree}: Heating Load Error = {heating_error}, Cooling Load Error = {cooling_error}')

    OutputFormat.print_divider('h2')

    optimal_heating_degree = np.argmin(heating_errors)
    optimal_cooling_degree = np.argmin(cooling_errors)

    print(f'Optimal Degree for Heating Load: {optimal_heating_degree}')
    print(f'Optimal Degree for Cooling Load: {optimal_cooling_degree}')

    return optimal_heating_degree, optimal_cooling_degree

########################################################################################################################
# Task 7:  Evaluation and Visualisation of Results
########################################################################################################################

def task7(df, optimal_heating_degree, optimal_cooling_degree):
    OutputFormat.print_header('h1', 'Task 7:  Evaluation and Visualisation of Results')

    estimate_and_plot(df, optimal_heating_degree, optimal_cooling_degree)

def estimate_and_plot(df, optimal_heating_degree, optimal_cooling_degree):
    labels, targets = separate_labels_and_targets(df)
    features = labels.values
    heating_targets = targets.iloc[:, 0].values
    cooling_targets = targets.iloc[:, 1].values

    # Estimate model parameters for heating loads
    heating_coefficients = fit_polynomial_model(optimal_heating_degree, features, heating_targets)
    predicted_heating_loads = polynomial_model(optimal_heating_degree, features, heating_coefficients)

    # Estimate model parameters for cooling loads
    cooling_coefficients = fit_polynomial_model(optimal_cooling_degree, features, cooling_targets)
    predicted_cooling_loads = polynomial_model(optimal_cooling_degree, features, cooling_coefficients)

    # Plot estimated vs true loads for heating
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.scatter(heating_targets, predicted_heating_loads)
    plt.plot([heating_targets.min(), heating_targets.max()], [heating_targets.min(), heating_targets.max()], 'r--')
    plt.xlabel('True Heating Loads')
    plt.ylabel('Estimated Heating Loads')
    plt.title('Heating Loads: True vs Estimated')

    # Plot estimated vs true loads for cooling
    plt.subplot(1, 2, 2)
    plt.scatter(cooling_targets, predicted_cooling_loads)
    plt.plot([cooling_targets.min(), cooling_targets.max()], [cooling_targets.min(), cooling_targets.max()], 'r--')
    plt.xlabel('True Cooling Loads')
    plt.ylabel('Estimated Cooling Loads')
    plt.title('Cooling Loads: True vs Estimated')

    plt.tight_layout()
    plt.show()

    # Calculate and output the mean absolute difference
    heating_mad = mean_absolute_error(heating_targets, predicted_heating_loads)
    cooling_mad = mean_absolute_error(cooling_targets, predicted_cooling_loads)

    print(f'Mean Absolute Difference for Heating Loads: {heating_mad}')
    print(f'Mean Absolute Difference for Cooling Loads: {cooling_mad}')

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

########################################################################################################################
# Main Function
########################################################################################################################

def main():
    OutputFormat.print_header('title', 'Machine Learning Assignment 3')

    start_time = time.time()

    original_data = pd.read_csv(source_filename + file_extension, header=0)

    labels, targets = task1(original_data.copy())
    optimal_heating_degree, optimal_cooling_degree = task6(labels, targets)
    task7(original_data.copy(), optimal_heating_degree, optimal_cooling_degree)

    print(f'\nTotal Runtime: {time.time() - start_time:.2f}s')

if __name__ == '__main__':
    main()