import warnings
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
warnings.filterwarnings("ignore", message="Reloaded modules: <module_name>")
import tensorflow as tf
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense
import time
from sklearn.metrics.pairwise import cosine_similarity
from math import *
from sklearn.preprocessing import MinMaxScaler

#***************************Data  Importation  *********************************

movies = pd.read_csv("u.item", 
                   engine='python', sep='|', 
                   names=["MovieID", "Title","Release","videorelease","IMDb URL",
                          "unknown", "Action","Adventure","Animation",
                          "Children\'s","Comedy","Crime","Documentary","Drama","Fantasy",
                          "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
                          "Thriller","War","Western"],
                   encoding='latin-1')

ratings = pd.read_csv("ratings.csv",
                      engine='python', sep=',')
                      
users = pd.read_csv("u.user", 
                   engine='python', sep='|', 
                   names=["UserID", "Age", "Gender","Occupation", "Zip-code"],
                   encoding='latin-1')

#***************************  Data Preprocessing *********************************

movies.drop(['Release', 'videorelease','IMDb URL'], axis=1, inplace=True)

movies.insert(21, 'Genres',"", allow_duplicates=False)
genre_columns = movies.columns[2:]
movies['Genres'] = movies[genre_columns].apply(lambda row: 
                                               [genre for genre in genre_columns
                                                if row[genre] == 1], axis=1)

movies.drop(['unknown', 'Action','Adventure','Animation',
'Children\'s','Comedy','Crime','Documentary','Drama','Fantasy',
'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi',
'Thriller','War','Western'], axis=1, inplace=True)

#delete rows with unknown genres
movies = movies.drop(movies[movies['Genres'].apply(lambda x: x == ['unknown'])].index)

ratings.rename(columns={'userId':'UserID','movieId':'MovieID',
                        'rating':'Rating','timestamp':'Timestamp'}, inplace=True)

# Age column preprocessing
age_bins = [0, 18, 24, 34, 44, 49, 55, float('inf')]
age_labels = [1, 18, 25, 35, 45, 50, 56]
users['Age'] = pd.cut(users['Age'], bins=age_bins, labels=age_labels)

# Gender column preprocessing
users["Gender"]= users["Gender"].replace("M",0) 
users["Gender"]= users["Gender"].replace("F",1) 

# Occupation column preprocessing
"""
occupation_mapping = {"administrator": 1,"artist": 2,"doctor": 3,"educator": 4,
    "engineer": 5,"entertainment": 6, "executive": 7,"healthcare": 8,
    "homemaker": 9, "lawyer": 10, "librarian": 11, "marketing": 12,"none": 13,
    "other": 14,"programmer": 15, "retired": 16, "salesman": 17, "scientist": 18,
    "student": 19, "technician": 20, "writer": 21}
"""
occupation_mapping = {"administrator": 3,"artist": 2,"doctor": 6,"educator": 1,
    "engineer": 17,"entertainment": 5, "executive": 7,"healthcare": 6,
    "homemaker": 9, "lawyer": 11, "librarian": 8, "marketing": 14,"none": 0,
    "other": 0,"programmer": 12, "retired": 13, "salesman": 14, "scientist": 15,
    "student": 4, "technician": 17, "writer": 20}

users["Occupation"] = users["Occupation"].map(occupation_mapping)

users.drop(['Zip-code'], axis=1, inplace=True) 

userscomparison=users.copy()

# Age column rescaling
scaler = MinMaxScaler()
users['Age'] = users['Age'].astype(float)
users['Age'] = scaler.fit_transform(users['Age'].values.reshape(-1,1))


usersratings = userscomparison.merge(ratings, how='inner')
usersratingsmovies=usersratings.merge(movies, how='inner')
usersratingsmovies.drop(['Timestamp', 'Title','Rating','MovieID','UserID'], 
                        axis=1, inplace=True) 

########################### DNN: inputs & outputs##############################
###############################################################################
Frames=usersratingsmovies.copy()
mlb = MultiLabelBinarizer()
Y=mlb.fit_transform(Frames['Genres'])

DATASET=Frames.values
X= DATASET[:,0:3].astype(float)


######################## Demographic component #################################
################################################################################
################################################################################

#********************************* DNN Model **********************************

XTrain,XTest,yTrain,yTest = model_selection.train_test_split(X,Y,test_size =0.2,
                                                             random_state=1)
modelMc = Sequential()
modelMc.add(Dense(units=100,input_dim=3,activation="relu"))
modelMc.add(Dense(units=100,activation="relu"))
modelMc.add(Dense(units=100,activation="relu"))
modelMc.add(Dense(units=100,activation="relu"))
modelMc.add(Dense(units=100,activation="relu"))
modelMc.add(Dense(units=18,activation="sigmoid"))
modelMc.compile(loss="binary_crossentropy",
                optimizer="adam",metrics=[
        'BinaryAccuracy',
        tf.keras.metrics.Precision(thresholds=0.25),
        tf.keras.metrics.Recall(thresholds=0.25)])


start_time = time.time()
history=modelMc.fit(XTrain,yTrain,
                    verbose=1,epochs=500,batch_size=100,
                    validation_split=0.2)

end_time = time.time()
training_time = end_time - start_time

predictions = modelMc.predict(XTest)
predictions[predictions >= 0.25] = 1
predictions[predictions < 0.25] = 0

######################## New User Demographic Profile #########################
###############################################################################
condition=True

while condition:
    Newusergender=int(input("New User's Gender[0:M,1:F]:"))
    Newuserage=int(input("New User's Age [1, 18, 25, 35, 45, 50, 56]:"))
    Newuseroccupation=int(input("New User's Occupation [0-20]:"))
    
    Newuser=[]
    Newuser.append(Newusergender)
    Newuser.append(Newuserage)
    Newuser.append(Newuseroccupation)
    Newuser = np.array(Newuser)
    Newuser=np.reshape(Newuser,(1,3))
    
    #***************************** Category* Prediction ***************************
    
    categories=modelMc.predict(Newuser)
    categorie = np.argmax(categories, axis=1)
    genres=mlb.classes_
    labelcategorie=genres[categorie[0]]
    CategoryEtoile=[]
    CategoryEtoile.append(labelcategorie)
    
      
    ######################## Collaborative component ###############################
    ################################################################################
    ################################################################################
    
    Movies=movies.copy()
    CategorieEtoileMovies= Movies[Movies['Genres'].apply(lambda x: CategoryEtoile[0] in x)]
    
    
    #********************** Category* Ratings datraframe ****************************
    
    CategoryEtoileRatings = ratings[ratings.MovieID.isin(CategorieEtoileMovies['MovieID'])]
      
    #********************** Simulate the cold start problem ****************************     
     
    selected_users = CategoryEtoileRatings.groupby('UserID').filter(lambda x: len(x) <= 5)
    users_with_more_than_5 = CategoryEtoileRatings.groupby('UserID').filter(lambda x: len(x) > 5)
    selected_users = selected_users.append(users_with_more_than_5.groupby('UserID')
                                           .apply(lambda x: x.sample(n=5, 
                                                                     random_state=42)))
    selected_users= selected_users.reset_index(drop=True)
    Ratings=selected_users
    
    
    usersratings1 = userscomparison.merge(Ratings, how='inner')
    usersratingsmovies1=usersratings1.merge(CategorieEtoileMovies, how='inner')
    NewuserIDs=usersratingsmovies1.query('Age==@Newuserage and Gender==@Newusergender and Occupation==@Newuseroccupation')['UserID']
    
    lst = NewuserIDs.tolist()
    
    if len(lst) > 0:
        user = lst[0]
        print(lst[0])
        condition=False
    else:
        print ("User Not Found")
        condition=True

#******************************* Collaborative Filtering **********************

Mean = Ratings.groupby(by="UserID",as_index=False)['Rating'].mean()

Rating_avg = pd.merge(Ratings,Mean,on='UserID')
Rating_avg['adg_rating']=Rating_avg['Rating_x']-Rating_avg['Rating_y']

final=pd.pivot_table(Rating_avg,values='adg_rating',index='UserID',columns='MovieID')

check = pd.pivot_table(Rating_avg,values='Rating_x',index='UserID',columns='MovieID')


#FONCTION GET USER SIMILAR MOVIES 
def get_user_similar_movies( user1, user2 ):
  common_movies = Ratings[Ratings.UserID == user1].merge(Ratings[Ratings.UserID == user2], on = "MovieID", how = "inner" )
  return common_movies.merge( CategorieEtoileMovies, on = 'MovieID' )

#FONCTION FIND N NEIGHBOURS 
def find_n_neighbours(df,n):
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:n].index, 
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df

#DEFINITION FONCTION USER ITEM SCORE
def User_item_score(user,item):
    a = sim_user_30_m[sim_user_30_m.index==user].values
    b = a.squeeze().tolist()
    c = final_movie.loc[:,item]
    d = c[c.index.isin(b)]
    f = d[d.notnull()]
    avg_user = Mean.loc[Mean['UserID'] == user,'Rating'].values[0]
    index = f.index.values.squeeze().tolist()
    corr = similarity_with_movie.loc[user,index]
    fin = pd.concat([f, corr], axis=1)
    fin.columns = ['adg_score','correlation']
    fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
    nume = fin['score'].sum()
    deno = fin['correlation'].sum()
    final_score = avg_user + (nume/deno)
    return final_score


Rating_avg = Rating_avg.astype({"MovieID": str})

Movie_user = Rating_avg.groupby(by = 'UserID')['MovieID'].apply(lambda x:','.join(x))

# Replacing NaN by Movie Average Rating
final_movie = final.fillna(final.mean(axis=0))

cosine = cosine_similarity(final_movie)
np.fill_diagonal(cosine, 0 )

similarity_with_movie =pd.DataFrame(cosine,index=final_movie.index)
similarity_with_movie.columns=final_movie.index

sim_user_30_m = find_n_neighbours(similarity_with_movie,20)


#********************************Predicted ratings***********************************
MoviesIDs=list(final_movie.columns)

RatingsCategory=pd.DataFrame(columns=['MovieID','PredictedRating'])

nb=0
for i in MoviesIDs:
    predicted_rating = User_item_score(user,i)
    RatingsCategory.loc[nb]=[int(i),predicted_rating]
    nb+=1


#*********** Movies Sentiment Analysis (creation des fichiers SA)***********************
#***************************************************************************************
#***************************************************************************************
#***************************************************************************************
"""
#*******Fonction sentiment_scores: Analyse des sentiments d'une phrase  **************
#*************************************************************************************
def sentiment_scores(sentence):
# Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
# polarity_scores method of SentimentIntensityAnalyzer 
# object gives a sentiment dictionary.
# which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)
    resNeg.append(sentiment_dict['neg'])
    resPos.append(sentiment_dict['pos'])
    resNeu.append(sentiment_dict['neu'])
    rescomp.append(sentiment_dict['compound'])
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        SA.append("Positive")
    elif sentiment_dict['compound'] <= - 0.05 :
        SA.append("Negative")
    else :
        SA.append("Neutral")

       
#***********************************lecture fichier***********************************
#*************************************************************************************


#Parcourir les fichiers du repertoire MoviesTweets
fichiers= listdir('MoviesTweets')
#print(fichiers)

for i in fichiers:
    print(i)
    if (i != 'desktop.ini') :
        #i='jihene'+name   
        #ouvrir le fichier en lecture
        file=i
        csvFile = open(file, 'r')
        
        #mettre les donnees dans un dataframe DFTweet
        DFTweet=pd.read_csv(csvFile,delimiter=',',names=["author_id","created_at","context_annotations","text"])
        
        #"text"Column Cleaning 
        
        from nltk.stem import LancasterStemmer
        lst = LancasterStemmer()
        from nltk.corpus import stopwords
        a = set(stopwords.words('english'))
        token=[]
        TextCleanedList=[]
        for text in DFTweet["text"]:
            TextCleaned=""
            token = word_tokenize(text.lower())
            stopwords = [x for x in token if x not in a]
            for m in stopwords:
                TextCleaned+=(lst.stem(m))+" "
            TextCleanedList.append(TextCleaned)    
            
        #Creer un nouveau dataframe columns=['author_id','text','text-nltk'] avec 'text-nltk'est le 'text' nettoye
        DRResult=DFTweet.loc[:,['author_id','text']]
        DRResult['text-nltk']=TextCleanedList
        #print(DRResult.head())
        
        # Analyse des valeurs null
        print(DFTweet.isnull().sum())
        
        
        #**************************Analyse de sentiments du 'text'***************************
        #*************************************************************************************
        
        resNeg=[]
        resPos=[]
        resNeu=[]
        rescomp=[]
        SA=[]
        for text in DRResult["text"]:
           sentiment_scores(text)
        DRResult["SentimentNegative"]=resNeg
        DRResult["SentimentPositive"]=resPos
        DRResult["SentimentNeutral"]=resNeu
        DRResult["SentimentCompound"]=rescomp
        DRResult["SentimentAnalysis"]=SA
        print(DRResult.head())
        
        #i=i[6:]
        titre='SA'+i[:len(i)-4]+'.xlsx'
        DRResult.to_excel(titre) 
        print('SentimentAnalysis is exported to Excel File successfully.')

"""


#************************Composante Sentiment Analysis*********************************
#***************************************************************************************
#***************************************************************************************
#***************************************************************************************

PolarityScoreCategory = pd.DataFrame(columns=['MovieID', 'Title', 'PolarityScore'])
cp = 0
for i,j in zip(CategorieEtoileMovies.Title, CategorieEtoileMovies.MovieID):
    f='SA'+i+'.xlsx'
    try:
        df=pd.read_excel(f)
    except FileNotFoundError:
        continue
    except OSError:
        continue
    else:            
        PolarityScoreCategory.loc[cp]=[j,i,df['SentimentCompound'].mean()]
        cp+=1

print('****************** Category* Polarity Score  *****************')
print(PolarityScoreCategory)   


#************************ Category* Ratings & Polarity Score*****************************

print('******************Category* Predicted Ratings & Polarity Score******************')
CategoryRatingsPolarityScore= PolarityScoreCategory.merge(RatingsCategory, how='inner')
pd.set_option('display.max_columns',None)
print(CategoryRatingsPolarityScore)

#************************ Category* Weighted Scores ************************************
#print('******************Category*  Weighted Scores ******************')

# Polar=CategoryRatingsPolarityScore['PolarityScore']
# Ratin=CategoryRatingsPolarityScore['PredictedRating']


#************************RS Evaluation RMSE*********************************
#***************************************************************************************
#***************************************************************************************
#*************************************************************

#***************** Movies from Category* rated by New user*************************
MoviesCategoryratedUser=Ratings[Ratings.UserID==user]
print(MoviesCategoryratedUser)

#***************** Movies Category* rated by New user & predicted scores *************************
MoviesCategoryratedUserScores= pd.merge(MoviesCategoryratedUser,CategoryRatingsPolarityScore,on='MovieID')
print(MoviesCategoryratedUserScores)

MoviesCategoryratedUserScores.drop(['Timestamp'], axis=1, inplace=True) 
MoviesCategoryratedUserScores.drop(['Title'], axis=1, inplace=True) 
print(MoviesCategoryratedUserScores)


#************** Programmation lineaire optimisation de la selection de alpha *******************************

import pulp as p 
for i in range(len(MoviesCategoryratedUserScores)):
    a= MoviesCategoryratedUserScores.iloc[i,3]
    b= MoviesCategoryratedUserScores.iloc[i,4]
    c= MoviesCategoryratedUserScores['Rating'].mean()
    #c= MoviesCategoryratedUserScores.iloc[i,2]
    
    Lp_prob = p.LpProblem('Problem', p.LpMinimize)  
    x = p.LpVariable("x", lowBound = 0,upBound=1,cat ='Continuous')   # Create a variable x >= 0
    y = p.LpVariable("y", lowBound = 0,upBound=1,cat ='Continuous')   # Create a variable y >= 0
    Lp_prob += - a * x - b * y + c 
    Lp_prob +=  x+y ==1
    Lp_prob +=  a * x + b * y>=1
    Lp_prob +=  a * x + b * y<=5
    
    #print(Lp_prob) 
    status = Lp_prob.solve()   
    #print(p.LpStatus[status]) 
    #print(p.value(x), p.value(Lp_prob.objective))
    alpha=p.value(x)
    beta=p.value(y)
    #beta=1- alpha 
    WeightedScore=alpha * a + beta * b
    WeightedScoreminusRating=p.value(Lp_prob.objective)
    MoviesCategoryratedUserScores.loc[i,'WeightedScore']= WeightedScore
    MoviesCategoryratedUserScores.loc[i,'TwitCoFiDError']=WeightedScoreminusRating*WeightedScoreminusRating
RR=MoviesCategoryratedUserScores['Rating']
#WS=MoviesCategoryratedUserScores['WeightedScore']
PR=MoviesCategoryratedUserScores['PredictedRating']


MoviesCategoryratedUserScores['DFCFError']=(RR-PR)*(RR-PR)
print(MoviesCategoryratedUserScores)


#****************************RMSE***********************************************
listeTwitCoFiD=list(MoviesCategoryratedUserScores['TwitCoFiDError'])
sommeTwitCoFiD=fsum(listeTwitCoFiD)
RMSETwitCoFiD=sqrt(sommeTwitCoFiD/len(MoviesCategoryratedUserScores))
print("RMSE TwitCoFiD: {:.4f}".format(RMSETwitCoFiD))

listeDFCF=list(MoviesCategoryratedUserScores['DFCFError'])
sommeDFCF=fsum(listeDFCF)
RMSEDFCF=sqrt(sommeDFCF/len(MoviesCategoryratedUserScores))
print("RMSE CoDFi-DL: {:.4f}".format(RMSEDFCF))

#****************************MAE***********************************************

def somme_valeurs_absolues(liste):
    somme = 0
    for element in liste:
        somme += abs(element)
    return somme

resultatTwitCoFiD = somme_valeurs_absolues(listeTwitCoFiD)
MAETwitCoFiD=resultatTwitCoFiD/len(listeTwitCoFiD)
print("MAE TwitCoFiD: {:.4f}".format(MAETwitCoFiD))


resultatCoDFiDL = somme_valeurs_absolues(listeDFCF)
MAECoDFiDL=resultatCoDFiDL/len(listeDFCF)
print("MAE CoDFi-DL: {:.4f}".format(MAECoDFiDL))







