#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

# Imports
from cProfile import run
from importlib.metadata import files
import io
import collections, functools, operator
import datetime as dt
from readline import set_completer_delims
import numpy as np
import random as r
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse import linalg

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
def parse_file(file, movies, users, ratings, limit_entries, limit_size):
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
                movies.append(last_movie)
                users.append(int(split_line[0]))
                ratings.append(int(split_line[1]))
                counter += 1


# Create two sparse matrices from the three lists (because the actual matrix would be too large for memory!).
# One that has movies as rows and users as columns, with the contents of the cells being the ratings given by the user for the movies, 
# And one that has users as columns and the movies as rows.
def create_sparse_matrices(movies, users, ratings):
    # TODO: better performance using np-arrays, why?
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

def taak3(matrix):
    def checkresults(org_matrix, new_matrix, unique_rows, file):
        with open(file + '.txt', 'w') as testfile:
            for r in unique_rows:
                testfile.write(str(r) + ':\n')
                row = org_matrix.getrow(r)
                cols = row.indices
                for c in cols:
                    testfile.write(str(c) + ',' + str(round(new_matrix[r,c])) + '\n')

    # Summarize dataset with SVD.
    k = min(matrix.shape[0] - 1, 5)
    print('k=' + str(k))
    q_matrix, s, vt = linalg.svds(matrix, k = k)
    
    # From SVD we can calculate the matrices Q and P, which are needed for the SGD algorithm.
    pt_matrix = diags(s) @ vt

    nonzero_rows, nonzero_cols = matrix.nonzero()
    unique_rows = np.unique(nonzero_rows)
    hyperparam_1, hyperparam_2 = 1, 1
    gradient_step = 0.00001
    epochs = 1

    checkresults(matrix, q_matrix @ pt_matrix, unique_rows, "before")

    for a in range(epochs):
        # Stochastic zou random moeten zijn, wij loopen er helemaal door, zou dat ok zijn?
        # TODO: shuffle de unique_rows
        for m in unique_rows:
            #print("Epoch " + str(a) + ": " + str(m))
            row = matrix.getrow(m)
            cols = row.indices
            
            for u in cols:
                #print("Epoch " + str(a) + "| movie: " + str(m) + " - user: " + str(u))

                nabla_q, nabla_p = 0, 0

                for f in range(k):
                    A = matrix[m,u]
                    B = q_matrix[m,:] @ pt_matrix[:,u]
                    C = pt_matrix[f,u]
                    D = q_matrix[m,f]

                    nabla_q += (-2 * (A - B) * C) + (2 * hyperparam_1 * D)
                    nabla_p += (-2 * (A - B) * D) + (2 * hyperparam_1 * C)
                
                correction_curr_col = gradient_step * nabla_p
                pt_matrix[:,u] = pt_matrix[:,u] - correction_curr_col

                correction_curr_row = gradient_step * nabla_q
                q_matrix[m,:] = q_matrix[m,:] - correction_curr_row

    checkresults(matrix, q_matrix @ pt_matrix, unique_rows, "after")

def plot():
    # X axis parameter:
    xaxis = np.array([2, 8])

    # Y axis parameter:
    yaxis = np.array([4, 9])

    plt.plot(xaxis, yaxis)
    plt.show()


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
        text += ": took " + str(hours) + "hours, " + str(minutes) + "minutes and " + str(seconds) + "seconds."

    print(text)
    return now


#####################
#    RUN SECTION    #
#####################
# Prepare the program for running task1, task2 & task3
movies, users, ratings = [], [], []
datasetFolder = "./netflix dataset/"
print("Current time - elapsed time since previous print")

# TASK 1 - Loading the dataset:
# Parameters:
limit_entries = False # Set this boolean to True to only read the first 'limit_size' amount of entries of each given file, set to False to read full files
limit_size = 10000

# Execution:
# Parse the following input files:
t1_start = get_and_print_time("Task 1", None) # retrieve (& print) current timestamp to measure task1 efficiency
parse_file(datasetFolder + "combined_data_1.txt", movies, users, ratings, limit_entries, limit_size)
parse_file(datasetFolder + "combined_data_2.txt", movies, users, ratings, limit_entries, limit_size)
parse_file(datasetFolder + "combined_data_3.txt", movies, users, ratings, limit_entries, limit_size)
parse_file(datasetFolder + "combined_data_4.txt", movies, users, ratings, limit_entries, limit_size)
t1_input_parsed = get_and_print_time("Finished reading input files", t1_start)

# Use the movies, users & ratings arrays to create two sparse matrices.
movies_x_users, users_x_movies = create_sparse_matrices(movies, users, ratings)
t1_input_parsed = get_and_print_time("Finished creating both sparse matrices", t1_input_parsed)
print()

# TASK 2 - DIMSUM
# Parameters:
gammas = [1] # run with one gamma
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
    actual_atraspose_a = movies_x_users @ users_x_movies
    t2_actual = get_and_print_time("Finished calculating actual A^T * A", t2_approximation)

    # Compare exact A^T * A with approximated one by calculating MSE
    mse = (np.square(np.subtract(actual_atraspose_a, approx_atranspose_a))).mean()
    t2_comparison = get_and_print_time("Finished calculating MSE of A^T & A", t2_actual)
    print("MSE for gamma " + str(gamma) + " was: " + str(mse)) # if MSE is closer to 0 => better approximation

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




get_and_print_time("Finished all tasks", t1_start)



# nonzero_rows, nonzero_cols = users_x_movies.nonzero()
# unq_nonzero_rows = np.unique(nonzero_rows)
# unq_nonzero_cols = np.unique(nonzero_cols)
# norms = calculate_vector_norm(users_x_movies, unq_nonzero_cols)
# t2_norms = get_and_print_time("Finished creating norms", t2_start)
# map_result = dimsum_mapper(users_x_movies, norms, unq_nonzero_rows, gamma)
# t2_mapper = get_and_print_time("Finished mapper", t2_norms)
# b_matrix = dimsum_reducer(map_result, norms, len(movies) + 1)
# t2_reducer = get_and_print_time("Finished reducer", t2_mapper)
# approx_atraspose_a = approximate_atranspose_a(b_matrix, norms, len(movies) + 1)
# t2_approximation = get_and_print_time("Finished approximation", t2_reducer)
# compare_atranspose_a(actual_atraspose_a, approx_atraspose_a)
# t2_compare = get_and_print_time("Finished comparison", t2_approximation)
# t2_finish = get_and_print_time("Finished task 2", t2_start)

#print()
#t3_start = get_and_print_time("Task 3", None)
#taak3(movies_x_users)
#t3_finish = get_and_print_time("Finished task 3", t3_start)


