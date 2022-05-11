#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

# Imports
import io
import collections, functools, operator
import datetime as dt
import numpy as np
import random as r
from scipy.sparse import csr_matrix, lil_matrix


# Parse an input file.
# Lines that contain a number and a colon indicate a MovieID.
# The following lines list, for that particular movie, UserIDs, 
# a comma, a rating, a comma, then the date when the rating was done.
def parse_file(file, movies, users, ratings):
    with io.open(file, "r") as f:
        last_movie = 0

        for line in f:
            if ":" in line:
                last_movie = int(line.split(":")[0])
            else:
                # TODO: remove
                if r.random() < 1:
                    split_line = line.split(",")
                    movies.append(last_movie)
                    users.append(int(split_line[0]))
                    ratings.append(int(split_line[1]))


# Create two sparse matrices from three lists.
# One that has movies rows and users columns, with the contents of the cells 
# being the ratings given by the user for the movies, and one that has users
# columns and movies rows.
def create_sparse_matrices(movies, users, ratings):
    # TODO: better performance using np-arrays, why?
    movies_array = np.asarray(movies)
    users_array = np.asarray(users)
    ratings_array = np.asarray(ratings)

    movies_x_users = csr_matrix((ratings_array, (movies_array, users_array)))

    # TODO: transpose should be better than performing the create_matrix function(?)
    users_x_movies = movies_x_users.transpose()

    return movies_x_users, users_x_movies


# Calculate the exact A transpose * A operation.
def calculate_atranspose_a(matrix_t, matrix):
    return matrix_t @ matrix


# Calculate the norms for a matrix given an array of columns which contain at least one value.
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
                # TODO: should we store multiplication of the norms?
                #       (Needed during the approximation of A transpose * A)
                probability = min(1, gamma / (norms[j] * norms[k]))
                if r.random() < probability:
                    emissions[(j, k)] = matrix[i, j] * matrix[i, k]

        return collections.Counter(emissions)

    return map(m, rows)


def dimsum_reducer(emissions, nr_of_movies):
    # def r(m):
    #     print(m)
    #     print(len(m))

    # functools.reduce(r, emissions)

    reducer_result = functools.reduce(operator.add, emissions)

    # #b_matrix = np.zeros(shape = (nr_of_movies, nr_of_movies))
    # b_matrix = lil_matrix((nr_of_movies, nr_of_movies))

    # for k, v in reducer_result.items():
    #     i = k[0]
    #     j = k[1]
    #     n = norms[i] * norms[j]

    #     if (gamma / n ) > 1:
    #         value = v / n
    #     else:
    #         value = v / gamma
        
    #     b_matrix[i, j] = b_matrix[i, j] + value
    
    # return b_matrix


def approximate_atranspose_a(b_matrix, norms, nr_of_movies):
    #d_matrix = np.zeros(shape = (nr_of_movies, nr_of_movies))
    d_matrix = lil_matrix((nr_of_movies, nr_of_movies))
    
    for n in norms:
        d_matrix[n, n] = norms[n]
    
    return d_matrix @ b_matrix @ d_matrix
    #return np.matmul(np.matmul(d_matrix, b_matrix), d_matrix)
    

def compare_atranspose_a(actual, approx):
    print("Comparing results")
    
    # print(actual.sorted_indices())
    # print()
    # print(approx.sorted_indices())

    # list = [1, 2]#[10, 15, 123, 45, 78]
    # for l in list:
    #     print("Real: " + str(actual[l,l]))
    #     print("Approx: " + str(approx[l,l]))


def get_and_print_time(msg, prev):
    # Om printen makkelijk terug te kunnen uitschakelen.
    if 1 == 1:
        now = dt.datetime.now()
        text = now.strftime("%H:%M:%S")
        if prev is not None:
            text += "  " + str(now - prev)
        if msg is not None:
            text += "  " + msg

# Program
def program():
    movies, users, ratings = [], [], []
        print(text)

        return now


# Run
movies, users, ratings = [], [], []
gamma = 100

print()

t1_start = get_and_print_time("Task 1", None)
#parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/smallest.txt", movies, users, ratings)
parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/small.txt", movies, users, ratings)
#parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/medium.txt", movies, users, ratings)
#parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_1.txt", movies, users, ratings)
#parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_2.txt", movies, users, ratings)
#parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_3.txt", movies, users, ratings)
#parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_4.txt", movies, users, ratings)
t1_read = get_and_print_time("Finished reading file", t1_start)
movies_x_users, users_x_movies = create_sparse_matrices(movies, users, ratings)
t1_sparse = get_and_print_time("Finished creating sparse matrices", t1_read)
t1_finish = get_and_print_time("Finished task 1", t1_start)

print()

t2_start = get_and_print_time("Task 2", None)
actual_atraspose_a = calculate_atranspose_a(movies_x_users, users_x_movies)
nonzero_rows, nonzero_cols = users_x_movies.nonzero()
unq_nonzero_rows = np.unique(nonzero_rows)
unq_nonzero_cols = np.unique(nonzero_cols)
norms = calculate_vector_norm(users_x_movies, unq_nonzero_cols)
t2_norms = get_and_print_time("Finished creating norms", t2_start)
map_result = dimsum_mapper(users_x_movies, norms, unq_nonzero_rows, gamma)
t2_mapper = get_and_print_time("Finished mapper", t2_norms)
b_matrix = dimsum_reducer(map_result, len(movies) + 1)
t2_reducer = get_and_print_time("Finished reducer", t2_mapper)
approx_atraspose_a = approximate_atranspose_a(b_matrix, norms, len(movies) + 1)
t2_approximation = get_and_print_time("Finished approximation", t2_reducer)
compare_atranspose_a(actual_atraspose_a, approx_atraspose_a)
t2_compare = get_and_print_time("Finished comparison", t2_approximation)
t2_finish = get_and_print_time("Finished task 2", t2_start)

print()
get_and_print_time("Finished project", t1_start)
