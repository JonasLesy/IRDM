#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

# Imports
import io
import datetime as dt
import numpy as np
import random
from scipy.sparse import csr_matrix


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
                split_line = line.split(",")
                movies.append(last_movie)
                users.append(int(split_line[0]))
                ratings.append(int(split_line[1]))


# Create two sparse matrices from three lists.
# One that has movies rows and users columns, with the contents of the cells 
# being the ratings given by the user for the movies, and one that has users
# columns and movies rows.
def create_sparse_matrices(create_matrix, movies, users, ratings):
    # TODO: better performance using np-arrays, why?
    movies_array = np.asarray(movies)
    users_array = np.asarray(users)
    ratings_array = np.asarray(ratings)

    movies_x_users = create_matrix((ratings_array, (movies_array, users_array)))
    # TODO: transpose should be better than performing the create_matrix function(?)
    users_x_movies = movies_x_users.transpose()

    return movies_x_users, users_x_movies


# Calculate the exact A transpose * A operation.
def calculate_atranspose_a(matrix_t, matrix):
    return matrix_t * matrix


# Calculate the norms for a matrix given an array of columns which contain at least one value.
def calculate_vector_norm(matrix, cols):
    dict = {}
    for c in cols:
        dict[c] = np.linalg.norm(matrix[:,c].toarray())
    
    return dict


def dimsum_mapper(matrix, norms, rows, cols, gamma):
    print("MAPPER")

    emissions = {}

    # TODO: do we need to calculate everything or should an upper triangular matrix be sufficient?
    # TODO: we need unique cols per row (not from the matrix). Result now contains zero values.
    for j in cols:
        for k in cols:
            # The probability is the same for every row
            probability = min(1, gamma / (norms[j] * norms[k]))

            for i in rows:
                if random.random() < probability:
                    emissions[(i, j, k)] = matrix[i, j] * matrix[i, k]

    print(emissions)


def dimsum_reducer():
    print("DIMSUM reducer")


def perform_dimsum(matrix, gamma):
    print("DIMSUM")

    nonzero_rows, nonzero_cols = matrix.nonzero()
    unq_nonzero_rows = np.unique(nonzero_rows)
    unq_nonzero_cols = np.unique(nonzero_cols)

    norms = calculate_vector_norm(matrix, unq_nonzero_cols)
    
    emissions = dimsum_mapper(matrix, norms, unq_nonzero_rows, unq_nonzero_cols, gamma)





# Program
def program():
    movies, users, ratings = [], [], []

    parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/sample.txt", movies, users, ratings)
    #parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_1.txt", movies, users, ratings)
    #parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_2.txt", movies, users, ratings)
    #parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_3.txt", movies, users, ratings)
    #parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_4.txt", movies, users, ratings)

    # Task 1
    # TODO: choice between csr en csc might impact other tasks.
    movies_x_users, users_x_movies = create_sparse_matrices(csr_matrix, movies, users, ratings)
    
    # Task 2
    #x = calculate_atranspose_a(movies_x_users, users_x_movies)
    #print(x)
    perform_dimsum(users_x_movies, 25)

    return movies_x_users, users_x_movies


# Run
start = dt.datetime.now()
movies_x_users, users_x_movies = program()
end = dt.datetime.now()

# Debug info
if 1 == 0:
    print("Start: " + start.strftime("%d/%m/%Y, %H:%M:%S"))
    print("Einde: " + end.strftime("%d/%m/%Y, %H:%M:%S"))
    diff = end - start    
    print("Duur: " + str(diff))
    
    #print("Nonzero movies_x_users: " + str(movies_x_users.count_nonzero()))
    #print("Nonzero users_x_movies: " + str(users_x_movies.count_nonzero()))
