##### Preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
%matplotlib inline

# Reading the data
movies_df = pd.read_csv('D:\\1\\ml-latest\\movies.csv')
ratings_df = pd.read_csv('D:\\1\\ml-latest\\ratings.csv')

# Removing the year from the title column and store in a new year column.
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand=False)
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)',expand=False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
# print(movies_df.head())

#Dropping the genres column
movies_df = movies_df.drop('genres', 1)
# print(movies_df.head())

#Drop removes timestamp from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)
# print(ratings_df.head())

##### Modeling: Collaborative Filtering
# Creating Input User to recommand movies to

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputMovies = pd.DataFrame(userInput)

# Add movieId to input user
inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
inputMovies = pd.merge(inputId, inputMovies)
inputMovies = inputMovies.drop('year', 1)
# print(inputMovies.head())

# Getting the users who has seen the same movies
userSubset = ratings_df[ratings_df['movieId'].isin(inputMovies['movieId'].tolist())]
# print(userSubset.head())

# Grouping up the rows by userId
userSubsetGroup = userSubset.groupby(['userId'])
# print(userSubsetGroup.get_group(13))

#Sorting it so users with movie most in common with the input will have priority
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
# print(userSubsetGroup[0:3])

# Similarity of users to input user
# Selecting a subset of users to iterate through
userSubsetGroup = userSubsetGroup[0:100]

# Calculating the Pearson Correlation between input user and subset group, and store it in a dictionary
# Store the Pearson Correlation in a dictionary
pearsonCorrelationDict = {}

for name, group in userSubsetGroup:
    # Sorting the input and current user group so the values aren't mixed up later on
    group = group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by='movieId')
    # Get the N for the formula
    nRatings = len(group)
    # Get the review scores for the movies that they both have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    # Storing them in a temporary buffer variable in a list format to facilitate future calculations
    tempRatingList = temp_df['rating'].tolist()
    # Placing the current user group reviews in a list format
    tempGroupList = group['rating'].tolist()
    # Calculating the pearson correlation between two users, so called, x and y
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    # If the denominator is different than zero, then divide, else, 0 correlation.
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0
# print(pearsonCorrelationDict.items())

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
# print(pearsonDF.head())

# The top x similar users to input user
topUsers = pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
# print(topUsers.head())

# Rating of selected users to all movies
topUsersRating = topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
# print(topUsersRating.head())

#Multiplies the similarity by the user's ratings
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
# print(topUsersRating.head())

#Applies a sum to the topUsers after grouping it up by userId
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
# print(tempTopUsersRating.head())

#Creates an empty dataframe
recommendation_df = pd.DataFrame()
#Now we take the weighted average
recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
# print(recommendation_df.head())

# Let's sort it and see the top 20 movies that the algorithm recommended!
recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
# print(recommendation_df.head(10))

movies_df.loc[movies_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]

