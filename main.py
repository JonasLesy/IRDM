#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

# Imports
from cProfile import run
from importlib.metadata import files
import io
import os
import collections, functools, operator
import datetime as dt
from readline import set_completer_delims
import numpy as np
import random as r
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix, diags, linalg

################
#    TASK 1    #
################

# Parse the given input file.
# Lines that contain a number and a colon indicate a MovieID.
# The following lines list, for that particular movie:
# UserIDs, a rating, the date when the rating was done.
# The inputs are the filepath, empty movies array, empty users array, empty ratings array and a boolean to decide to limit the read entries to 200.
# The method will read the input file and fill the three arrays accordingly, the date has no further use in this project
#
# The following entry rating of score 4 by user with ID 9999 for movie with ID 50 will be stored in 3 the arrays as follows:
#            ----------------
# movies =    ... | 50 | ...
#            ----------------
#            ----------------
# users  =   ... | 9999 | ...
#            ----------------
#            ----------------
# ratings =    ... | 4 | ...
#            ----------------
#                    ^ all have the same index in their respective array
def parse_file(file, movies, users, ratings, limit_entries, limit_size, testing_dict):
    with io.open(file, "r") as f:
        last_movie = 0
        counter = 0

        for line in f:
            if limit_entries and counter == limit_size: # break the input parsing if enough items are read
                break
            if ":" in line: # the line is a movieID line
                last_movie = int(line.split(":")[0]) 
            else: # the line is a rating
                split_line = line.split(",")
                userID = int(split_line[0])
                rating = int(split_line[1])
                if (last_movie, userID) in testing_dict:
                    movies_testing.append(last_movie) # put testing sample in global testing array, didn't add this to function parameters for clarity
                    users_testing.append(userID)
                    ratings_testing.append(rating)
                else:
                    movies.append(last_movie)
                    users.append(userID)
                    ratings.append(rating)
                    counter += 1


# Create two sparse matrices from the three lists (because the actual matrix would be too large for memory!).
# One that has movies as rows and users as columns, with the contents of the cells being the ratings given by the user for the movies, 
# And one that has users as columns and the movies as rows.
def create_sparse_matrices(movies, users, ratings):
    movies_array = np.asarray(movies)
    users_array = np.asarray(users)
    ratings_array = np.asarray(ratings)

    # Create a sparse matrix with the rows representing the movies and the columns representing the users
    movies_x_users = csr_matrix((ratings_array, (movies_array, users_array)), dtype=float)

    # Create a sparse matrix with the rows representing the users and the columns representing the movies
    # Not used anymore because of transposition optimization (see line below)
    #users_x_movies = csr_matrix((ratings_array, (users_array, movies_array)), dtype=float)
    users_x_movies = movies_x_users.transpose()

    return movies_x_users, users_x_movies




################
#    TASK 2    #
################

# Calculate the norms for a matrix given an array of columns which contain at least one value.
# This method outputs a dictionary for easy retrieval of the column norms
def calculate_vector_norm(matrix, cols):
    dict = {}
    for c in cols:
        dict[c] = np.linalg.norm(matrix[:,c].toarray())

    return dict


def dimsum_mapper(matrix, norms, rows, gamma):

    def m(i):
        row = matrix.getrow(i)
        cols = row.indices
        emissions = {}

        for j in cols:
            for k in cols:
                if r.random() < min(1, gamma / (norms[j] * norms[k])):
                    emissions[(j, k)] = matrix[i, j] * matrix[i, k]
        return collections.Counter(emissions)
    
    return map(m, rows)


def dimsum_reducer(emissions, norms, nr_of_movies):
    reducer_result = functools.reduce(operator.add, emissions)

    b_matrix = lil_matrix((nr_of_movies, nr_of_movies))

    for k, v in reducer_result.items():
        i = k[0]
        j = k[1]
        n = norms[i] * norms[j]

        if (gamma / n ) > 1:
            value = v / n
        else:
            value = v / gamma
        
        b_matrix[i, j] = b_matrix[i, j] + value

    return b_matrix


def approximate_atranspose_a(b_matrix, norms, nr_of_movies):
    d_matrix = lil_matrix((nr_of_movies, nr_of_movies))
    
    for n in norms:
        d_matrix[n, n] = norms[n]
    
    return d_matrix @ b_matrix @ d_matrix





################
#    TASK 3    #
################

# TODO: commentaar
def calculate_nabla_q_and_p(ratings, q_matrix, pt_matrix, update_nabla, hyperparam_1, hyperparam_2):
    nonzero_rows, nonzero_cols = ratings.nonzero()
    unique_rows = np.unique(nonzero_rows)
    np.random.shuffle(unique_rows)

    for i in unique_rows: # improve Q and Ptranspose row by row, Q and P have the same dimensions, so this can be done in the same loop
        row = ratings.getrow(i)
        cols = row.indices
        np.random.shuffle(cols)

        for x in cols:
            known_rating = ratings[i, x] # Rix in formula

            for f in range(k):
                q_value = q_matrix[i,f] # Qif in formula
                p_value = pt_matrix[f,x] # Pxf in formula

                update_nabla(i, 
                             x,
                             f, 
                             (-2 * (known_rating - (q_value * p_value)) * p_value) + (2 * hyperparam_2 * q_value),
                             (-2 * (known_rating - (p_value * q_value)) * q_value) + (2 * hyperparam_1 * p_value))


# TODO: commentaar
def batch_gradient_descent(ratings, q_matrix, pt_matrix, gradient_step, hyperparam_1, hyperparam_2):
    nabla_Q_matrix = lil_matrix((q_matrix.shape[0], q_matrix.shape[1]), dtype=float)    # generate a new matrix to store the nabla q's in
    nabla_P_matrix = lil_matrix((pt_matrix.shape[0], pt_matrix.shape[1]), dtype=float)  # generate a new matrix to store the nabla p's in
    
    def update_nabla(i, x, f, nabla_q, nabla_p):
        nabla_Q_matrix[i, f] += (gradient_step * nabla_q)
        nabla_P_matrix[f, x] += (gradient_step * nabla_p)

    calculate_nabla_q_and_p(ratings, q_matrix, pt_matrix, update_nabla, hyperparam_1, hyperparam_2)

    return np.subtract(q_matrix, nabla_Q_matrix), np.subtract(pt_matrix, nabla_P_matrix)


# TODO: commentaar
def stochastic_gradient_descent(ratings, q_matrix, pt_matrix, gradient_step, hyperparam_1, hyperparam_2):
    def update_nabla(i, x, f, nabla_q, nabla_p):
        q_matrix[i, f] -= (gradient_step * nabla_q)
        pt_matrix[f, x] -= (gradient_step * nabla_p)

    calculate_nabla_q_and_p(ratings, q_matrix, pt_matrix, update_nabla, hyperparam_1, hyperparam_2)


def calculate_accuracy(original_matrix, q_matrix, pt_matrix):
    nonzero_rows, nonzero_cols = original_matrix.nonzero()
    unique_rows = np.unique(nonzero_rows)
    total_sum = 0
    m_matrix = q_matrix @ pt_matrix
    m_rows, m_cols = m_matrix.shape[0], m_matrix.shape[1]
    for x in unique_rows:
        # accomodate for the testing dataset containing higher dimensions, not used in training, this can happen because we can limit the training sampling to X lines per file
        # This will never happen if limit_entries = False
        if x <= m_rows: 
            row = original_matrix.getrow(x)
            cols = row.indices
            for i in cols:
                if i <= m_cols:
                    total_sum += (m_matrix[x, i] - original_matrix[x, i]) ** 2

    return np.sqrt(total_sum) / original_matrix.nnz


########################
#    HELPER METHODS    #
########################

# This method returns the current time, prints the given message,
# and calculates the difference if a previous timestamp is provided
def get_and_print_time(msg, prev):
    now = dt.datetime.now()
    text = now.strftime("%H:%M:%S")
    if msg is not None:
        text += "  " + msg
    if prev is not None:
        timedelta = now - prev # add the difference between now and previous to the message, if previous was given
        hours, remainder = divmod(timedelta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        text += ": took " + str(hours) + " hours, " + str(minutes) + " minutes and " + str(seconds) + " seconds."

    print(text)
    return now

def parse_probe_file(file, testing_dict):
    with io.open(file, "r") as f:
        last_movie = 0

        for line in f:
            if ":" in line: # the line is a movieID line
                last_movie = int(line.split(":")[0]) 
            else: # the line is a userID
                userID = int(line.split("\n")[0])
                testing_dict[(last_movie, userID)] = True


#####################
#    RUN SECTION    #
#####################
# Prepare the program for running task1, task2 & task3
movies, users, ratings = [], [], []
datasetFolder = os.path.dirname(__file__) + "/netflix dataset/"
print("Current time - elapsed time since previous print")

# TASK 1 - Loading the dataset:
# Parameters:
limit_entries = True # Set this boolean to True to only read the first 'limit_size' amount of entries of each given file, set to False to read full files
limit_size = 1000
split_training = True

# Execution:
# Parse the following input files:
t1_start = get_and_print_time("Task 1", None) # retrieve (& print) current timestamp to measure task1 efficiency
testing_dict = {}
movies_testing, users_testing, ratings_testing = [], [], []
if split_training: # If you choose to split training and testing data, a probe.txt will be read to define the testing dataset
    parse_probe_file(datasetFolder + "probe.txt", testing_dict)
    t1_start = get_and_print_time("Finished reading probe file", t1_start)
parse_file(datasetFolder + "combined_data_1.txt", movies, users, ratings, limit_entries, limit_size, testing_dict)
parse_file(datasetFolder + "combined_data_2.txt", movies, users, ratings, limit_entries, limit_size, testing_dict)
parse_file(datasetFolder + "combined_data_3.txt", movies, users, ratings, limit_entries, limit_size, testing_dict)
parse_file(datasetFolder + "combined_data_4.txt", movies, users, ratings, limit_entries, limit_size, testing_dict)
t1_input_parsed = get_and_print_time("Finished reading input files", t1_start)

# Use the movies, users & ratings arrays to create two sparse matrices.
movies_x_users, users_x_movies = create_sparse_matrices(movies, users, ratings)
t1_input_matrices_made = get_and_print_time("Finished creating both sparse matrices", t1_input_parsed)

if split_training:
    # Here we don't use the create_sparse_matrices function used above, as we only need movies_x_users for our task 3 and 4 implementation
    # if task2 required testing and training split, we should also use the create_sparse_matrices function (not part of the assignment)
    movies_array = np.asarray(movies_testing)
    users_array = np.asarray(users_testing)
    ratings_array = np.asarray(ratings_testing)
    # Create a sparse matrix with the rows representing the movies and the columns representing the users
    movies_x_users_testing = csr_matrix((ratings_array, (movies_array, users_array)), dtype=float)
    t1_input_matrices_made = get_and_print_time("Finished creating sparse matrix for testing", t1_input_matrices_made)

print()

# TASK 2 - DIMSUM
# Parameters:
gammas = []
#gammas = [1] # run with one gamma
#gammas = [0.01, 0.1, 1, 2, 5, 10, 50, 100, 500, 1000, 5000] # run with > 1 gamma

# Execution: (in for loop to possibly run with different gammas)
MSEs = [] # keep track of calculated MSEs of each gamma
runtimes = [] # keep track of the runtime for each gamma
for gamma in gammas:
    ts_start = dt.datetime.now() # additional timestamp to calculate current loop execution time
    t2_start = get_and_print_time("Task 2", None)

    # Calculate the matrix's norms beforehand
    nonzero_rows, nonzero_cols = users_x_movies.nonzero()
    unq_nonzero_cols = np.unique(nonzero_cols)
    unq_nonzero_rows = np.unique(nonzero_rows)
    norms = calculate_vector_norm(users_x_movies, unq_nonzero_cols)
    t2_norms = get_and_print_time("Finished calculating norms", t2_start)

    # Run the DIMSUM mapper
    # Here we run the mapper in a single thread on a single machine, this could be executed in a distributed computing setting as well.
    # For example, by distributing the rows to different compute nodes.
    map_result = dimsum_mapper(users_x_movies, norms, unq_nonzero_rows, gamma) # This delives a dictionary of all emitted pairs
    t2_mapper = get_and_print_time("Finished mapper", t2_norms)

    # Run the DIMSUM reducer
    # Same as with the mapper, parts of the dictionary of all emitted pairs could be distributed to different compute nodes
    b_matrix = dimsum_reducer(map_result, norms, max(movies) + 1)
    t2_reducer = get_and_print_time("Finished reducer", t2_mapper)

    # Compute approximation of A^T * A using b_matrix
    approx_atranspose_a = approximate_atranspose_a(b_matrix, norms, max(movies) + 1)
    t2_approximation = get_and_print_time("Finished approximation of A^T * A", t2_reducer)

    # Calculate the exact A^T * A, with A = users_x_movies
    actual_atranspose_a = movies_x_users @ users_x_movies
    t2_actual = get_and_print_time("Finished calculating actual A^T * A", t2_approximation)

    # Compare exact A^T * A with approximated one by calculating MSE
    mse = (np.square(np.subtract(actual_atranspose_a, approx_atranspose_a))).mean()
    t2_comparison = get_and_print_time("Finished calculating MSE of A^T & A", t2_actual)
    get_and_print_time("MSE for gamma " + str(gamma) + " was: " + str(mse), None) # if MSE is closer to 0 => better approximation

    MSEs.append(mse)
    loop_runtime = dt.datetime.now() - ts_start
    runtimes.append(loop_runtime.total_seconds())

if(len(gammas) > 1): # only plot charts if the program ran for more than one gamma
    figure, (mseChart, runtimeChart) = plt.subplots(1, 2)
    figure.suptitle('MSE VS. Runtime')
    mseChart.plot(gammas, MSEs)
    runtimeChart.plot(gammas, runtimes)
    mseChart.set_ylabel('MSE')
    mseChart.set_xlabel('gamma')
    runtimeChart.set_ylabel('Runtime (in seconds)')
    runtimeChart.set_xlabel('gamma')
    plt.show()

print()

# TASK 3 - (Stochastic) Gradient Descent with Latent Factors
# Parameters:
epochs = 20 # Control the number of epochs to execute
matrix_shape = movies_x_users.shape
#k = max(1, (min(matrix_shape[0], matrix_shape[1]) - 1)) # Control the number of eigenvalues to be used in SVD, rule: 1 <= k <= kmax, with kmax is the smallest dimension of the matrix minus one
k = 6
stochastic_gradient_step = 0.00001 # learning rate for stochastic gradient descent
batch_gradient_step = 0.1 # learning rate for batch gradient descent
hyperparam_1, hyperparam_2 = 1, 1 # user set regularization parameters to accommodate for scarcity, can be used to shrink aggressively where data are scarce

# Execution:
t3_start = get_and_print_time("Task 3", None)

# Summarize dataset with SVD.
q_matrix, s, vtranspose = linalg.svds(movies_x_users, k = k)
q_matrix = csr_matrix(q_matrix) # make q_matrix sparse

# From SVD we can calculate the matrices Q and P, which are needed for the SGD algorithm.
ptranspose_matrix = diags(s) @ vtranspose # sigma is a diagonal matrix, so we only have to multiply those with vtranspose
ptranspose_matrix = csr_matrix(ptranspose_matrix) # make q_matrix sparse

q_matrix_for_BGD = q_matrix.copy() # take a copy of both Q and P so we can run the SGD and BGD loops simultaneously
ptranspose_matrix_for_BGD = ptranspose_matrix.copy()

get_and_print_time("Finished Q and P calculation", t3_start)

RMSEs_SGD = [] # keep track of calculated RMSEs of each gamma for SGD
RMSEs_BGD = [] # keep track of calculated RMSEs of each gamma for BGD
epochs_list = []

# Now iteratively improve q & p to accomodate for missing values in movies_x_users (=A)
for i in range(epochs):
    epochs_list.append(i + 1)
    t_epoch_start = get_and_print_time(None, None)

    # run a full epoch of SGD:
    stochastic_gradient_descent(movies_x_users, q_matrix, ptranspose_matrix, stochastic_gradient_step, hyperparam_1, hyperparam_2)
    t_sgd_end = get_and_print_time("Finished epoch " + str(i+1) + " of stochastic gradient descent", t_epoch_start)

    # calculate RMSE of SGD
    matrix_to_use_for_accurracy = movies_x_users
    if split_training:
        matrix_to_use_for_accurracy = movies_x_users_testing
    
    rmse_sgd = calculate_accuracy(matrix_to_use_for_accurracy, q_matrix, ptranspose_matrix)
    RMSEs_SGD.append(rmse_sgd)
    t_rmse_sgd = get_and_print_time("Calculated RMSE for SGD: " + str(rmse_sgd), t_sgd_end)

    #run a full epoch of BGD
    q_matrix_for_BGD, ptranspose_matrix_for_BGD = batch_gradient_descent(movies_x_users, q_matrix_for_BGD, ptranspose_matrix_for_BGD, batch_gradient_step, hyperparam_1, hyperparam_2)
    t_bgd_end = get_and_print_time("Finished epoch " + str(i+1) + " of batch gradient descent", t_rmse_sgd)

    # calculate RMSE of BGD
    rmse_bgd = calculate_accuracy(matrix_to_use_for_accurracy, q_matrix_for_BGD, ptranspose_matrix_for_BGD)
    RMSEs_BGD.append(rmse_bgd)
    t_rmse_bgd = get_and_print_time("Calculated RMSE for BGD: " + str(rmse_bgd), t_bgd_end)

if(len(epochs_list) > 1):
    figure, (sgdChart, bgdChart) = plt.subplots(1, 2)
    figure.suptitle('SGD VS. BGD - RMSE')
    sgdChart.plot(epochs_list, RMSEs_SGD)
    bgdChart.plot(epochs_list, RMSEs_BGD)
    sgdChart.set_ylabel('RMSE - SGD')
    sgdChart.set_xlabel('epoch')
    bgdChart.set_ylabel('RMSE - BGD')
    bgdChart.set_xlabel('epoch')
    plt.show()


get_and_print_time("Finished all tasks", t1_start)



