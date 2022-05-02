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
import time
import io

print("Welcome to the Information Retrieval & Data Mining assignment")
print("Made by Jonas Lesy & Thomas Bytebier")

start_time = time.time()

movieIDs, userIDs, ratings = [], [], []

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


#parse_file('./netflix dataset/combined_data_smaller.txt')
parse_file('./netflix dataset/combined_data_1million.txt', movieIDs, userIDs, ratings)
#parse_file('./netflix dataset/combined_data_1.txt', movieIDs, userIDs, ratings)
#parse_file('./netflix dataset/combined_data_2.txt', movieIDs, userIDs, ratings)
#parse_file('./netflix dataset/combined_data_3.txt', movieIDs, userIDs, ratings)
#parse_file('./netflix dataset/combined_data_4.txt', movieIDs, userIDs, ratings)
#parse_file('./netflix dataset/combined_data_1.txt')
#parse_file('./netflix dataset/combined_data_2.txt')
#parse_file('./netflix dataset/combined_data_3.txt')
#parse_file('./netflix dataset/combined_data_4.txt')

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

# For performance boost
ratings = np.asarray(ratings)
movieIDs = np.asarray(movieIDs)
userIDs = np.asarray(userIDs)
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
#print(userRowsMovieColumns)
#print("\n\n")

print("--- Task 1 took %s seconds ---" % (time.time() - start_time))

print("\n")

#################
#  DIMSUM CODE  #

"""
print(myaccessiblething.shape)
print(myaccessiblething.ndim)
print(myaccessiblething.nnz)
print(myaccessiblething.indices)
print(myaccessiblething.indptr)
print("blub")
"""
gamma = 500

# Calculate columnNorm: linalg.norm(userRowsMovieColumns.getcol(movieIdx)), e.g. movieID 1 = first el
# column norm for movieID 1 = linalg.norm(userRowsMovieColumns.getcol(1))

start_time = time.time()

#columnNorms = []
#for i in range(max(movieIDs)+1):
#    columnNorms.append(linalg.norm(userRowsMovieColumns.getcol(i)))

nonzero_rows, nonzero_cols = userRowsMovieColumns.nonzero()
unq_nonzero_rows = np.unique(nonzero_rows)
unq_nonzero_cols = np.unique(nonzero_cols)

columnNorms = {}
for c in unq_nonzero_cols:
    columnNorms[c] = linalg.norm(userRowsMovieColumns[:,c])


print("Highest norms = ", max(columnNorms))
  
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


#print(userIDs)
print("Got column norms")
print(len(columnNorms))

print("--- Column norms took %s seconds ---" % (time.time() - start_time))


    #mysum = 0
    #for user in userIDs:
    #    print("user is " + str(user))
    #    myaccessiblething.indptr[user]
        #print(user)
        #mysum += (myaccessiblething.data[user] ** 2)
    #columnNorms.append(math.sqrt(mysum))
    

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
    #print("indices are ", currentRow.indices)

    emissions = {}

    # double loop through row to get pairs Aij & Aik:
    for iterationJ, movieJ in enumerate(currentRow.indices):
        for iterationK, movieK in enumerate(currentRow.indices):
            if random.random() < min(1, gamma / (columnNorms[movieJ] * columnNorms[movieK])):
                emissions[(movieJ, movieK)] = userRowsMovieColumns[user, movieJ] * userRowsMovieColumns[user, movieK]

    return emissions

def dimsumReduce(emission, Bmatrix):
    #print("Got emission ", emission)
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
    numberOfMovies = max(columnNorms) + 1
    Dmatrix = np.zeros(shape = (numberOfMovies, numberOfMovies))
    for key in columnNorms:
        Dmatrix[key, key] = columnNorms[key]
    return Dmatrix
    
print("done")
Dmatrix = getMatrixDFromNorms(columnNorms)
#print(Dmatrix)

numberOfMovies = np.amax(movieIDs) + 1 # account for 0 column
Bmatrix = np.zeros(shape = (numberOfMovies, numberOfMovies))
print("Made Bmatrix: ")
#print(Bmatrix)
print("Running mapreduce...")
for user in np.unique(userIDs):
    mapEmissions = dimsumMap(user)
    #print("emission is ", mapEmissions)
    dimsumReduce(mapEmissions, Bmatrix)

#print(Bmatrix)
#print(Dmatrix)

print("Done!")

#print(Bmatrix)
#print(Dmatrix)
approximatedAtransposeA = np.matmul(np.matmul(Dmatrix, Bmatrix), Dmatrix)
#print(approximatedAtransposeA)

print("Calculating real A^T * A...")
realAtransposeA = userRowsMovieColumns.transpose().dot(userRowsMovieColumns)
#print(realAtransposeA.toarray())
print("...done!")

# Calculate Mean Squared Error (MSE)
# formula: MSE = 1 / n * SUM(actual - forecast)^2
print("Calculating MSE...")
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

print("--- Task 2 took %s seconds ---" % (time.time() - start_time))

"""

############
#  TASK 3  #
############
print('\n\n')
print("############\n", "#  TASK 3  #\n", "############\n\n")


print(movieRowsUserColumns)
movieRowsUserColumns = coo_matrix((ratings, (movieIDs, userIDs)), dtype=float)
movieRowsUserColumns = movieRowsUserColumns.tocsc()
print(movieRowsUserColumns.toarray())
print(movieRowsUserColumns.shape[0],movieRowsUserColumns.shape[1])

# get svd summarization
matrixU, Sigma, Vt = linalg.svds(movieRowsUserColumns, k = (min(movieRowsUserColumns.shape[0],movieRowsUserColumns.shape[1]) - 2))
print("U is")
print(np.matrix.round(matrixU, 3))
print("Sigma is")
print(np.matrix.round(Sigma, 3))
print("Vt is")
print(np.matrix.round(Vt, 3))

# R = movieRowsUserColumns
# Q = matrixU
# Pt = Sigma * Vt

# R ~= Q * Pt
# => to approximate an element we don't know, we need to get that element from the R matrix
# e.g. element 2,5 = take the product of all elements on row 2 in Q and all elements on column 5 in Pt and sum all those products = R[2,5]

"""

"""
mymatrix = coo_matrix(np.matrix([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [0, 0, 0, 0, 0], [0, 2, 0, 0, 0]]), dtype=float)
print(mymatrix)
matrixU, Sigma, Vt = linalg.svds(mymatrix.tocsr(), k = 3)
print("U is")
print(matrixU)
print("Sigma is")
print(Sigma)
print("Vt is")
print(Vt)
"""