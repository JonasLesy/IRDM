import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import coo_array
from scipy.sparse import csr_array
from scipy.sparse import linalg
import random
import itertools
import math

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

parse_file('./netflix dataset/combined_data_smallest.txt')

#print(movieIDs)
#print(userIDs)
#print(ratings)
#print("\n\n")
#sparse.coo_matrix((data, (row_ind, col_ind)))
#A = np.array(movieIDs, userIDs, ratings)
#movieIDArray = np.array(movieIDs)
#userIDArray = np.array(userIDs)
#row = np.array(movieIDs)
#col = np.array(userIDs)
movieRowsUserColumns = coo_matrix((ratings, (movieIDs, userIDs)))
movieRowsUserColumns = movieRowsUserColumns.tocsr()
userRowsMovieColumns = coo_matrix((ratings, (userIDs, movieIDs)))
userRowsMovieColumns = userRowsMovieColumns.tocsr()


print("Sparse matrix movies x users is:")
#print(movieRowsUserColumns)
print("\n\n")
sparse_mxu_matrix_size = movieRowsUserColumns.data.size/(1024**2)
print('Size of sparse MxU coo_matrix: '+ '%3.2f' %sparse_mxu_matrix_size + ' MB')
sparse_uxm_matrix_size = userRowsMovieColumns.data.size/(1024**2)
print('Size of sparse UxM coo_matrix: '+ '%3.2f' %sparse_uxm_matrix_size + ' MB')
print("Sparse matrix users x movies is:")
print(userRowsMovieColumns)
#print("\n\n")

print("\n")

#################
#  DIMSUM CODE  #
#################
mycustomthing = coo_matrix((ratings, (userIDs, movieIDs)))
myaccessiblething = mycustomthing.tocsr()
print('Custom:')
print(mycustomthing)
print('hello')
print(myaccessiblething.nonzero())
print(myaccessiblething.shape)
print(myaccessiblething.ndim)
print(myaccessiblething.nnz)
print(myaccessiblething.indices)
print(myaccessiblething.indptr)
print("blub")
gamma = 50

# Calculate columnNorm: linalg.norm(userRowsMovieColumns.getcol(movieIdx)), e.g. movieID 1 = first el
# column norm for movieID 1 = linalg.norm(userRowsMovieColumns.getcol(1))

for row in zip(userRowsMovieColumns.nonzero()):
    print(row)

columnNorms = []
for i in range(len(movieIDs)):
    columnNorms.append(linalg.norm(userRowsMovieColumns.getcol(i)))
    #print(userRowsMovieColumns.getcol(i))
    #print("movie on index " + str(i) + " is " + str(movieIDs[i]))


    
    #print(user)
    #numberOfReviewsByPerson = userRowsMovieColumns.getrow(user).nnz
    #currentRow = userRowsMovieColumns.getrow(user)
    #print('row was')
    #print(currentRow)
    #print("number of reviews is")
    #print(numberOfReviewsByPerson)
    #currentRow.indices = [1, 3] = MOVIEIDS
    #currentRow.data = [3, 4] = RATINGS

    #emissions = {}
    # double loop through row:
    #for iterationJ, movieJ in enumerate(currentRow.indices):
    #    for iterationK, movieK in enumerate(currentRow.indices):
    #        inverseColumnProbability = gamma / (columnNorms[movieJ] * columnNorms[movieK])
    #        print("inverseColumnProbability is", inverseColumnProbability)
    #        if random.random() < min(1, inverseColumnProbability):
    #            emissions[(iterationJ, iterationK)] = userRowsMovieColumns[user, movieJ] * userRowsMovieColumns[user, movieK]

    #print(emissions)

# userRowsMovieColumns.getrow(1000).data


print(userIDs)
print(columnNorms)
    #mysum = 0
    #for user in userIDs:
    #    print("user is " + str(user))
    #    myaccessiblething.indptr[user]
        #print(user)
        #mysum += (myaccessiblething.data[user] ** 2)
    #columnNorms.append(math.sqrt(mysum))
    
print("Column norms")
print(columnNorms)

print("Printing userentries")
#for i in range(0, userRowsMovieColumns.shape[0]):
#    print(userRowsMovieColumns.getrow(i))
#for i in range(len(userIDs)):
#    for m in range(len(movieIDs)):
#        print(i, myaccessiblething.indices[m], myaccessiblething.data[m])


# Map function gets a row, will possibly emit the product of any pair Aij Aik
def dimsumMap(i):
    currentRow = userRowsMovieColumns.getrow(i)
    #currentRow.indices = [1, 3] = MOVIEIDS
    #currentRow.data = [3, 4] = RATINGS

    emissions = {}

    # double loop through row to get pairs Aij & Aik:
    for iterationJ, movieJ in enumerate(currentRow.indices):
        for iterationK, movieK in enumerate(currentRow.indices):
            if random.random() < min(1, gamma / (columnNorms[movieJ] * columnNorms[movieK])):
                print("inverseColumnProbability for j ", iterationJ, " k ", iterationK, " is", 1 / (columnNorms[movieJ] * columnNorms[movieK]))
                
                emissions[(movieJ, movieK)] = userRowsMovieColumns[user, movieJ] * userRowsMovieColumns[user, movieK]

    return emissions

def dimsumReduce(emission, Bmatrix):
    print("Got emission ", emission)
    for element in emission:
        (aij, aik) = element
        inverseColNorms = 1 / (columnNorms[aij] * columnNorms[aik])
        if (gamma * inverseColNorms) > 1:
            valueToAdd = inverseColNorms * emission[element]
        else:
            valueToAdd = (1 / gamma) * emission[element]
        
        Bmatrix[aij, aik] = Bmatrix[aij, aik] + valueToAdd
        #(aij, aik) = element
        #print("Got element ", emission[element], "on position ", aij, " and ", aik)

def getMatrixDFromNorms(columnNorms):
    numberOfMovies = len(columnNorms)
    Dmatrix = np.zeros(shape = (numberOfMovies, numberOfMovies))
    np.fill_diagonal(Dmatrix, columnNorms)
    return Dmatrix
    
print("done")

numberOfMovies = np.amax(movieIDs) + 1 # account for 0 column
Bmatrix = np.zeros(shape = (numberOfMovies, numberOfMovies))
print(Bmatrix)
for user in np.unique(userIDs):
    mapEmissions = dimsumMap(user)
    dimsumReduce(mapEmissions, Bmatrix)

def calculateRealAtransposeA(sparse_matrix):
    transpose = sparse_matrix.transpose()
    return transpose

Dmatrix = getMatrixDFromNorms(columnNorms)
print(Bmatrix)
print(Dmatrix)
approximatedAtransposeA = Dmatrix * Bmatrix * Dmatrix
print(approximatedAtransposeA)

realAtransposeA = userRowsMovieColumns.transpose().dot(userRowsMovieColumns)
print(realAtransposeA.toarray())

# Calculate Mean Squared Error (MSE)
# formula: MSE = 1 / n * SUM(actual - forecast)^2
mse = (np.square(realAtransposeA - approximatedAtransposeA)).mean(axis=None)
print("Mean Squared Error is: ", mse)




# for all rows
#for i in range(0, myaccessiblething.shape[0]):
    #print(myaccessiblething.getrow(i).toarray()[0])


#for user in userIDs:
#    print("User was" + str(user))
#    for indj in range(myaccessiblething.indptr[user], myaccessiblething.indptr[user+1]):
#        print(user, myaccessiblething.indices[indj], myaccessiblething.data[indj])
#        for indk in range(myaccessiblething.indptr[user], myaccessiblething.indptr[user+1]):

#            inverseProbabilityOfColumnNorms = 1 #/ (columnNormJ * columnNormK)
            #print(min(1, gamma * inverseProbabilityOfColumnNorms))
            #print(user, myaccessiblething.indices[indj], myaccessiblething.data[indj])
# Loop over all rows
# for each row, combine all pairs & roll a die to see if they are to be emitted or not (popular columns are less probable to be emitted)


#def dimsum(A, gamma):
#    noRows = userRowsMovieColumns.get_shape()[0]
#    noCols = userRowsMovieColumns.get_shape()[1]
#    i = int(0)
#    while (i < noRows):
#        rowOnI = userRowsMovieColumns.getrow(i)
#        print("Got: " + rowOnI)
#        i += 1

#dimsum(userRowsMovieColumns, "")
#for row in zip(userRowsMovieColumns.row):
#    print("{0}".format(row))

#print(userRowsMovieColumns.getrow(0))


# ||X|| = De wortel nemen van de som van de kwadraten van de elementen van X

# Map over matrix Ri:
# for all pairs (Aij, Aik) in Ri do
# with probability   min(1, gamma * (1/||Cj||*||Ck||)
# emit ((Cj, Ck) -> (Aij, Aik))


# Reduce results:
# if gamma / (||Ci||*||Cj||) > 1 then Bij = 1 / (||Ci||*||Cj||) * SUM from i = 1 to R of Vi   = KNOWN VALUES (?)
# else Bij = (1 / gamma) * SUM from i = 1 to R of Vi   = APPROXIMATIOn (?)


# Step 1: map A on B (mxm matrix) with B having cosine similarities

# Step 2: compute approximation of A^T A by using B, see Chapter 4 slide 118

# Step 3: also compute exact A^T A by using A with SciPy.sparse library

# Step 4: compare our approximation with calculated one by using MSE & compare values