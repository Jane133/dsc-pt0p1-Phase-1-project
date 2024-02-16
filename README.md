# Your code here - remember to use markdown cells for comments as well!

# Import standard packages
import pandas as pd
import numpy as np

# import
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

#loading data
df_title_ratings = pd.read_csv("zippedData/imdb.title.ratings.csv.gz",)
df_title_basics = pd.read_csv("zippedData/imdb.title.basics.csv.gz",)
df_movie_gross = pd.read_csv("zippedData/bom.movie_gross.csv.gz",)
df_movies= pd.read_csv("zippedData/tmdb.movies.csv.gz",index_col=0)

#accessing the first 5 rows in df_title_basics
df_title_basics.head()

#Checking metadata of df_title_basics
df_title_basics.info()

#accessing the last 5 rows in df_title_basics
df_title_basics.tail()

#Checking out the rows and columns of df_title_basics
df_title_basics.shape

# checking the data types
df_title_basics.dtypes

#Accessing columns of df_title_basics
df_title_basics.columns

#Accessing the index of df_title_basics
df_title_basics.index

#aceesing the first 5 rows in df_title_ratings
df_title_ratings.head()

#Accessing concise summary of df_title_ratings

df_title_ratings.info()

#Accessing the last 3 rows of df_title_ratings.tail
df_title_ratings.tail(3)

#accessing description of df_movie_gross
df_title_ratings.describe()

#Analyzing rows and columns of df_title_ratings
df_title_ratings.shape

#Checking the data type
df_title_ratings.dtypes

#Accessing columns of df_title_ratings
df_title_ratings.columns

#Accessing the index of df_title_ratings
df_title_ratings.index

#Analyzing metadata of df_movie_gross.info

df_movie_gross.info()

#checking the first 5 rows of df_movie_gross
df_movie_gross.head()

#checking the last 5 rows of df_movie_gross
df_movie_gross.tail()

#print numbers pf rows and columns in bom_movie_gross_df
print(df_movie_gross.shape)

#Accessing the columns of df_movie_gross
print(df_movie_gross.columns)

print(df_movie_gross.dtypes)

#Accessing the index of df_movie_gross
print(df_movie_gross.index)

#Removing leading and trailing whitespace in df_title_basics columns
[col.strip() for col in df_title_basics.columns]

#Checking out the missing values in df_title_basics
missing_values = df_title_basics.isna().sum() 
missing_values

#Removing leading and trailing whitespace in 
[col.strip() for col in df_movie_gross.columns]

#Checking out the missing values in df_bom_movie_gross
missing_values = df_movie_gross.isna().sum()
missing_values

df_movies.info()

df_movies.head()

df_movies.tail()

df_movies.shape

df_movies.describe()

df_movies.dtypes

df_movies.index

#Removing leading and trailing whitespace in df_title_ratings columns
[col.strip() for col in df_title_ratings.columns]


#Calculating the percentage of missing values per column with respect to the entering df a function
def missing_values(data):
    """A simple function to identify data with missing values"""
    #identify the totsl missing values per column
    #sort in order
    miss = data.isnull().sum().sort_values(ascending = False)
    
    #calculate percentage of missing values
    percentage_miss = (data.isnull().sum() / len(data)).sort_values(ascending = False)
    
    #store in a dataframe
    missing = pd.DataFrame({"Missing Values": miss, "Percentage": percentage_miss}).reset_index()
    
    return missing
#applying function to the df_title_basics
missing_data = missing_values(df_title_basics)
missing_data 


#genres and original_title missing values are few hence drop them without causing any effect on data

#Filtering df that only contains Nan and empty strings
missing_genres = df_title_basics.genres.isna()
df_title_basics = df_title_basics[~missing_genres]


missing_title = df_title_basics.original_title.isna()
df_title_basics = df_title_basics[~missing_title]

#Confirming whether Nan values in original_title and genres columns have all dropped
missing_values(df_title_basics)

#Confirming whether Nan values in genres columns have all dropped

missing_values(df_title_basics)

#Visualization of runtime minutes

runtimes = df_title_basics.runtime_minutes

#finding max and mim runtime
min_runtime = runtimes.min()
max_runtime = runtimes.max()
mean_runtime = runtimes.mean()
print(f"minimum runtime: {min_runtime}")
print(f"Maximum runtime: {max_runtime}")
print(f"Mean runtime: {mean_runtime}")

#Choosing boxplot column
col_data = df_title_basics.runtime_minutes

# Creating boxplot
plt.figure(figsize=(12,6))
sns.set_context('notebook')
sns.boxplot(x= col_data, color= "yellow")
plt.title('Runtime minutes Boxplot');


#Checking for the highest runtime
df_title_basics.loc[df_title_basics.runtime_minutes == max_runtime]

sns.pairplot(df_movie_gross)

#replacing the missing values with  median runtime
df_title_basics.runtime_minutes.fillna(df_title_basics.runtime_minutes.median(), inplace=True)

#checking for residual of missing values in DF
missing_values

#cheking duplicates
duplicate_rows = df_title_basics.duplicated().sum()
print(f"Num of Duplicated Rows: {duplicate_rows}")


#Dealing with df_movie_gross
missing_data = missing_values(df_movie_gross)
missing_data

#Filtering dataframe to remove missing values in df_movie_gross
missing_studio = df_movie_gross.studio.isna()
df_movie_gross = df_movie_gross[~missing_studio]


#checking the residual of missing values
missing_values(df_movie_gross)

#removing unwanted characters like commas etc
df_movie_gross.foreign_gross.replace(',','', inplace=True, regex=True)

df_movie_gross.foreign_gross = df_movie_gross.foreign_gross.astype('float64')

# checking dtype success
df_movie_gross.dtypes

#Performing Visualization of df_movie_gross
income_foreign = df_movie_gross.foreign_gross
# find minimum and maximum values in  runtime_minutes coumn
min_foreign = income_foreign.min()
max_foreign = income_foreign.max()
mean_foreign = income_foreign.mean()
print(f"minimum runtime: {min_foreign}")
print(f"Maximum runtime: {max_foreign}")
print(f"Mean runtime: {mean_foreign}")
# selecting the column for the boxplot
col_data = df_movie_gross.foreign_gross
# Creating boxplot using Seaborn Library
plt.figure(figsize= (12,8))
sns.boxplot(x= col_data, color= "blue")
plt.title('Foreign Gross Income')
plt.xlabel('foreign_gross ')

#Replacing the missing vlaue with median
df_movie_gross.foreign_gross.fillna(df_movie_gross.foreign_gross.isna().median, inplace=True)

#Checking whether the performance was successful
missing_values(df_movie_gross)


min_domestic = income_domestic.min()
max_domestic = income_domestic.max()
mean_domestic = income_domestic.mean()
print(f"minimum runtime: {min_domestic}")
print(f"Maximum runtime: {max_domestic}")
print(f"Mean runtime: {mean_domestic}")

# selecting the column for the boxplot
col_data = df_movie_gross.domestic_gross

# Creating boxplot & histogram using Seaborn Library
fig, ax = plt.subplots(ncols=2, nrows=1, figsize= (15,5))
sns.boxplot(x=col_data, ax=ax[0], color='yellow')
ax[0].set_title('Domestic Income')
ax[0].set_xlabel('domestic_gross')
df_movie_gross.domestic_gross.hist(ax=ax[1], color='blue')
ax[1].set_title('Histogram ')
ax[1].set_xlabel('domestic_gross )')
plt.tight_layout;


#Repacing the missing values with Median
df_movie_gross.domestic_gross.fillna(df_movie_gross.domestic_gross.isna().median(), inplace=True)

#Verifying that the operation was successful 
#function & checking for any instances of duplicates
print(f"Num of duplicates: {df_movie_gross.duplicated().sum()}")
print(missing_values(df_movie_gross))


#Feature Engineering
#Merging different dataset in a single dataset
combined_data = df_movie_gross.merge(df_title_basics, left_on='title', right_on='original_title', how='left')
combined_data = combined_data.merge(df_title_ratings, left_on='tconst', right_on='tconst', how='left')
combined_data.head()

#Checking the missing value in the comnined data
missing_values(combined_data)


combined_data.shape

#Accessing Columns of interest and reassingning to df__movies
df_movies = df_movies.loc[:, ['original_title', 'vote_average', 'vote_count', 'release_date']]


#Checking whether the needed data is successfully extracted
df_movies.head()

# Convert 'release_date' column to date_time if it's not already in datetime format
df_movies['release_date'] = pd.to_datetime(df_movies['release_date'])

# Extract the year from 'release_date' and create a new column 'release_year'
df_movies['release_year'] = df_movies['release_date'].dt.year

# Drop the 'release_date' column if you want to remove it
df_movies.drop('release_date', axis=1, inplace=True)

# # Display the first few rows of the modified DataFrame
print(df_movies.head())


 #Executing an inner join to preserve only the movies appearing in both Tables
combined_data = combined_data.merge(df_movies, left_on= 'title', right_on = 'original_title', how="inner")
combined_data.head()


 missing_values(combined_data)


 # dropping the missing velues in start_year 
combined_data.drop('start_year', inplace=True, axis=1)

#Filling missing values in averagerating column with vote_ average column values
combined_data.averagerating.fillna(combined_data.vote_average, inplace=True)

#dropping vote_average columns
combined_data.drop('vote_average', axis=1, inplace=True)

#dropping vote_count columns and refilling the null values in num_votes with median

combined_data.drop('vote_count', inplace=True, axis=1)

#checking whether the vote_count colum has dropped successfully
combined_data.head()


 missing_values(combined_data)


 #Visualizing the distribution in combined dataframe for runtime_mintues
col_data = combined_data.runtime_minutes
plt.figure(figsize=(12,6))
sns.boxplot(x=col_data)
plt.title(' runtime_minutes');


#The data contains outliers
#Using imputation to replace the missing values

combined_data.runtime_minutes.fillna(combined_data.runtime_minutes.mean(),inplace=True)

#confirming if the operation was successful
missing_values(combined_data)



#visualizing the distribution of numvotes
print(combined_data.numvotes.agg(['mean', 'std', 'min', 'max']))
combined_data.numvotes.hist(color='skyblue', figsize=(10, 5))
plt.title("numvotes")
plt.xlabel('num_votes ')
plt.ylabel('frequency')



#Filling the skewed data using imputation median
combined_data.numvotes.fillna(combined_data.numvotes.median(), inplace=True)



missing_values(combined_data)


#Checking the most dominant genre using the process

#Calling the value_counts method to find the mode genre
top_genre = combined_data.genres.value_counts().head(1)
print(f'The most common genre in our data is: {top_genre}')


#Replacing the missing values with mode imputation
combined_data.genres.fillna(combined_data.genres.mode().iloc[0], inplace=True)


missing_values(combined_data)

#Checking the most dominant genre using the process

#Calling the value_counts method to find the mode genre
top_genre = combined_data.genres.value_counts().head(1)
print(f'The most common genre in our data is: {top_genre}')


#Replacing the missing values with mode imputation
combined_data.genres.fillna(combined_data.genres.mode().iloc[0], inplace=True)


 #confirming that we have no missing values in our dataset
missing_values(combined_data)


#Confirming the dtypes of combined dataset
combined_data.dtypes

#creating a new column that we will use to determine the financia success of the company
combined_data['total_gross'] = combined_data.domestic_gross.astype(str) + combined_data.foreign_gross.astype(str)

#determinng whether new column was created successfully
combined_data.head()

combined_data['total_gross']

#Splitting movie genres and exploded to allow for genre-specific analyses
combined_data.genres = combined_data.genres.str.split(',')
combined_data.head()

 #Exploding the genres
combined_data = combined_data.explode('genres')

 combined_data.head()

 # Plotting the domestic gross for each year
plt.figure(figsize=(10, 6))
plt.plot(combined_data['year'], combined_data['domestic_gross'], marker='o', color='skyblue', linestyle='-')

# Adding labels and title
plt.xlabel('Year')
plt.ylabel('Domestic Gross ($)')
plt.title('Domestic Gross for Each Year')

# Adding grid
plt.grid(True, linestyle='--', alpha=0.7)

# Show plot
plt.show()




# Creating a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = combined_data.corr()
plt.imshow(correlation_matrix, cmap= 'coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation = 45)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title("Correlation Heatmap")
plt.show()



 # Visualizing the relationship between genres and average ratings
plt.figure(figsize=(12, 6))
genre_avg_votes = combined_data.groupby('genres')['numvotes'].mean()
sorted_data = genre_avg_votes.sort_values(ascending=False)
colors = sns.color_palette("Set2", n_colors=len(sorted_data))
sorted_data.plot(kind='bar', color=colors)
plt.title("genres Number of Votes ")
plt.xlabel("vote_count")


# Visualizing the relationship between genre and average ratings
plt.figure(figsize=(12, 6))
genre_avg_ratings = combined_data.groupby('genres')['averagerating'].mean()
sorted_data = genre_avg_ratings.sort_values(ascending=False)
sorted_data.plot(kind='bar', color='green')
plt.title("Average Ratings ")
plt.xticks(rotation=90)
plt.xlabel("Genre")
plt.ylabel("Average Rating");

