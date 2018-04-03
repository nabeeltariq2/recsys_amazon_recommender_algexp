# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 04:26:18 2018

@author: ksiamionava3
"""

import pandas as pd
import gzip
import pandas as pd
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)
    
    
def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df_metadata = getDF('C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/meta_Digital_Music.json.gz')
rating = getDF('C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/reviews_Digital_Music_5.json.gz')

rating.to_csv('C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/reviews_Digital_Music.csv', sep=',')

df_metadata.to_csv('C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/meta_Digital_Music.csv', sep=',')

df = pd.read_csv("C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/meta_Digital_Music.csv",low_memory=False)
df_rating = pd.read_csv("C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/reviews_Digital_Music.csv",low_memory=False)

print(df_rating.columns)
print(df.columns)
print(df.shape)

df.head()

### Step1 genres transform delete all symbols
df["categories"]=map(lambda foo: foo.replace("[", ""), df["categories"])
df["categories"]=map(lambda foo: foo.replace("]", ""), df["categories"])
df["categories"]=map(lambda foo: foo.replace("-", ""), df["categories"])
df["categories"] = map(lambda foo: foo.replace("&", ""), df["categories"])
df["categories"] = map(lambda foo: foo.replace(" ", ""), df["categories"])
df["categories"]=map(lambda foo: foo.replace("'", ""), df["categories"])
df["categories"] = map(lambda foo: foo.lower(), df["categories"])

## Step 2 merge two datasets
dfMerged = pd.merge(df, df_rating, how='right', on=['asin'])
dfMerged

####create a unique simple ID for a user
dfMerged.reviewerID = dfMerged.reviewerID.astype("category")

dfMerged["userid_key"] = dfMerged["reviewerID"].cat.codes
dfMerged["userid_key"] = dfMerged["userid_key"] + 1

#Step 2 Derive  preferences
#group product genres checked by a user
UserGenreSummary = dfMerged.groupby("userid_key")["categories"].apply(list)
UserGenreSummary  = pd.DataFrame(UserGenreSummary)
UserGenreSummary.to_csv("C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/ProductGenreSummary.csv")

df3 = pd.read_csv("C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/ProductGenreSummary.csv")
df3 = df3.reset_index()

##Step 3 create the countvectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from pandas import DataFrame 
genres = df3["categories"] 
countVector = CountVectorizer(max_features = 500, stop_words='english') 
transformedcategories = countVector.fit_transform(genres) 


dfcat = DataFrame(transformedcategories.A, columns=countVector2.get_feature_names())
dfcat = dfcat.astype(int)

dfcat.head()
columns= countVector.get_feature_names()
#Step 4 repeat for the product genre table
PGenreSummary = df.groupby("asin")["categories"].apply(list)
PGenreSummary  = pd.DataFrame(PGenreSummary)
PGenreSummary.to_csv("C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/PGenreSummary.csv")
df4=pd.read_csv("C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/PGenreSummary.csv")
df4 = df4.reset_index()
genres2 = df4["categories"] 
countVector2 = CountVectorizer(max_features = 500, stop_words='english') 
transformedcategories2 = countVector2.fit_transform(genres2)
dfcatprod = DataFrame(transformedcategories2.A, columns=countVector2.get_feature_names())
dfcatprod = dfcatprod.astype(int)
columns=countVector2.get_feature_names()

dfcatprod.to_csv("C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/PGenrematrix.csv")
df5=pd.read_csv("C:/Users/ksiamionava3/OneDrive - Georgia Institute of Technology/Maching learning 2/PGenrematrix.csv")


# we convert music genres to a set of dummy variables 
df5[df5>=1] = 1


#in production you would use np.dot instead of writing your own dot product function.
def dot_product(vector_1, vector_2):  
    return sum([ i*j for i,j in zip(vector_1, vector_2)])

def get_movie_score(movie_features, user_preferences):  
    return dot_product(movie_features, user_preferences)

product1_features = df5.loc[0][columns]  
product1_features

user_preferences=dfcat.loc[0][columns]  
user_preferences

#predict importnance of the album for a user
prod1_user_predicted_score = dot_product(product1_features, user_preferences)  
prod1_user_predicted_score


##KNN
import numpy as np
X = np.array(df5)
 # create train and test
tpercent = 0.8
tsize = int(np.floor(tpercent * len(df5)))
dfReviews_train = X[:tsize]
dfReviews_test = X[tsize:]
#len of train and test
lentrain = len(dfReviews_train)
lentest = len(dfReviews_test)



from sklearn.neighbors import NearestNeighbors
neighbor = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(dfReviews_train)

#Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = neighbor.kneighbors(dfReviews_train)

#find most related products
for i in range(lentest):
    a = neighbor.kneighbors([dfReviews_test[i]])
    related_product_list = a[1]

    first_related_product = [item[0] for item in related_product_list]
    first_related_product = str(first_related_product).strip('[]')
    first_related_product = int(first_related_product)
    second_related_product = [item[1] for item in related_product_list]
    second_related_product = str(second_related_product).strip('[]')
    second_related_product = int(second_related_product)
    
    print ("Based on product reviews, for ", df3["asin"][lentrain + i] )
    print ("The first similar product is ", df3["asin"][first_related_product] )
    print ("The second similar product is ", df3["asin"][second_related_product] )
    print ("-----------------------------------------------------------")



#############
