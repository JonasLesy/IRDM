import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

print("Welcome to the Information Retrieval & Data Mining assignment")
print("Made by Jonas Lesy & Thomas Bytebier")


movieIDs = []
userIDs = []
ratings = []

# To read the input files, we're going to use regular expressions to make a distinction
# between a movieID line & a line that contains a rating given by a user
rx_dict = {
    'movieID': re.compile(r'(?P<movieID>(\d+)):\n'),
    'rating': re.compile(r'(?P<userID>(\d+)),(?P<rating>(\d)),(?P<date>(\d){4}-(\d){2}-(\d){2})(\n)?')
}

def _parse_line(line):
    """
    Do a regex search against all defined regexes and
    return the key and match result of the first matching regex

    """

    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            return key, match
    # if there are no matches
    return None, None

def parse_file(filename):
    with open(filename, 'r') as file_object:
        line = file_object.readline()
        while line:
            # at each line check for a match with a regex
            key, match = _parse_line(line)

            # extract school name
            if key == 'movieID':
                movieID = int(match.group('movieID'))

            # extract grade
            if key == 'rating':
                rating = int(match.group('rating'))
                userID = int(match.group('userID'))
                movieIDs.append(movieID)
                userIDs.append(userID)
                ratings.append(rating)
                #print('Rating for movie with id ' + movieID + ' is ' + rating + ' given by user ' + userID)
            
            
            line = file_object.readline()
    return

parse_file('./netflix dataset/combined_data_1.txt')

#print(movieIDs)
#print(userIDs)
#print(ratings)
#print("\n\n")
#sparse.coo_matrix((data, (row_ind, col_ind)))
#A = np.array(movieIDs, userIDs, ratings)
#movieIDArray = np.array(movieIDs)
#userIDArray = np.array(userIDs)
row = np.array(movieIDs)
col = np.array(userIDs)
movieRowsUserColumns = coo_matrix((ratings, (movieIDs, userIDs)))
userRowsMovieColumns = coo_matrix((ratings, (userIDs, movieIDs)))
print("Sparse matrix movies x users is:")
#print(movieRowsUserColumns)
print("\n\n")
sparse_mxu_matrix_size = movieRowsUserColumns.data.size/(1024**2)
print('Size of sparse MxU coo_matrix: '+ '%3.2f' %sparse_mxu_matrix_size + ' MB')
sparse_uxm_matrix_size = userRowsMovieColumns.data.size/(1024**2)
print('Size of sparse UxM coo_matrix: '+ '%3.2f' %sparse_uxm_matrix_size + ' MB')
#print("Sparse matrix users x movies is:")
#print(userRowsMovieColumns)
#print("\n\n")