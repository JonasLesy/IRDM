#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

# Imports
import io
import datetime as dt
import numpy as np
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
def create_sparse_matrix(movies, users, ratings):
    movies_array = np.asarray(movies)
    users_array = np.asarray(users)
    ratings_array = np.asarray(ratings)

    # TODO: choice between csr en csc might impact other tasks.
    movies_x_users = csr_matrix((ratings_array, (movies_array, users_array)))
    users_x_movies = csr_matrix((ratings_array, (users_array, movies_array)))

    return movies_x_users, users_x_movies
    

# Program
def program():
    movies, users, ratings = [], [], []

    #parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/sample.txt", movies, users, ratings)
    parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_1.txt", movies, users, ratings)
    #parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_2.txt", movies, users, ratings)
    #parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_3.txt", movies, users, ratings)
    #parse_file("/Users/thomasbytebier/Documents/School/IRDM/Project/netflix_dataset/combined_data_4.txt", movies, users, ratings)

    return create_sparse_matrix(movies, users, ratings)
    

# Run
start = dt.datetime.now()
movies_x_users, users_x_movies = program()
end = dt.datetime.now()

# Debug info
print("Start: " + start.strftime("%d/%m/%Y, %H:%M:%S"))
print("Einde: " + end.strftime("%d/%m/%Y, %H:%M:%S"))
diff = end - start    
print("Duur: " + str(diff))
print("Nonzero movies_x_users: " + str(movies_x_users.count_nonzero()))
print("Nonzero users_x_movies: " + str(users_x_movies.count_nonzero()))
