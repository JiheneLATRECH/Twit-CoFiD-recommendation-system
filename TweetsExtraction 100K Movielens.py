# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:52:19 2022
@author: Jihene LATRECH
"""

import tweepy
import datetime as dt
import pandas as pd
import csv
#import schedule
import time
from pathlib import Path
from csv import writer


API_key="LM8I79DBTXhkYpLay5yytMbsk"
API_keysecret="W63YYd5tr5ofUPlbDMPRu0alSOTfaa8Sq6HDtdODcAruPhaSLs"
bearer="AAAAAAAAAAAAAAAAAAAAACyBZgEAAAAAN74laP2reJod9Syv17NI%2FAuQnhw%3D2OAxbJlfmEZ6wjeWUybks9Nvf7liqWDsHoaMmrtpzqBBw11do2"
AppId="23494956"
consumer_id="pNXh6PZ8izkXRmPlhmx6O1Apx"
password="HNVk4g8SEkwZFEh3k9BVyFbEH5riwo6GoHnPapVLGNts2t8Bnp"
access_token="1493215286186713088-Upl1DvUsw9TOZ5IKZoSJS5Nzbw22nQ"
access_passwd="7v2hPCxvhTAGgKyj2E9XDDX7dlq7hwrUkiNnghOpKOllp"

#*************************** Data Importation *********************************

#print("dataframe movies") 
movies = pd.read_csv("u.item", 
                   engine='python', sep='|', 
                   names=["MovieID", "Title","Release","videorelease","IMDb URL",
                          "unknown", "Action","Adventure","Animation",
                          "Children\'s","Comedy","Crime","Documentary","Drama","Fantasy",
                          "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
                          "Thriller","War","Western"],encoding='latin-1')

movies.drop(['Release', 'videorelease','IMDb URL'], axis=1, inplace=True)


#print(movies.columns)
movies.insert(21, 'Genres',"", allow_duplicates=False)
print(movies.head())
listegenres=[]
for i in range (len(movies)):
    genres=[]
    if movies.iloc[i,2]==1:
        genres.append("unknown")
    if movies.iloc[i,3]==1:
        genres.append("Action")
    if movies.iloc[i,4]==1:
        genres.append("Adventure")
    if movies.iloc[i,5]==1:
        genres.append("Animation") 
    if movies.iloc[i,6]==1:
        genres.append("Children\'s")
    if movies.iloc[i,7]==1:
        genres.append("Comedy")
    if movies.iloc[i,8]==1:
        genres.append("Crime")
    if movies.iloc[i,9]==1:
        genres.append("Documentary") 
    if movies.iloc[i,10]==1:
        genres.append("Drama")
    if movies.iloc[i,11]==1:
        genres.append("Fantasy")
    if movies.iloc[i,12]==1:
        genres.append("Film-Noir")
    if movies.iloc[i,13]==1:
        genres.append("Horror") 
    if movies.iloc[i,14]==1:
        genres.append("Musical")
    if movies.iloc[i,15]==1:
        genres.append("Mystery")
    if movies.iloc[i,16]==1:
        genres.append("Romance") 
    if movies.iloc[i,17]==1:
       genres.append("Sci-Fi")
    if movies.iloc[i,18]==1:
       genres.append("Thriller")
    if movies.iloc[i,19]==1:
       genres.append("War") 
    if movies.iloc[i,20]==1:
       genres.append("Western") 
    movies.at[i,'Genres'] = genres


movies.drop(['unknown', 'Action','Adventure','Animation',
'Children\'s','Comedy','Crime','Documentary','Drama','Fantasy',
'Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi',
'Thriller','War','Western'], axis=1, inplace=True)

print(movies.head())


#suppression des films dont le titre comporte un caractere special : {':,?,/,*}
movies.drop(movies.loc[movies['Title']=='Naked Gun 33 1/3: The Final Insult (1994)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='What\'s Love Got to Do with It? (1993)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Venice/Venice (1992)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Shall We Dance? (1937)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Victor/Victoria (1982)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='JLG/JLG - autoportrait de décembre (1994)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='8 1/2 (1963)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='M*A*S*H (1970)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='B*A*P*S (1997)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Shall We Dance? (Shall We Dansu?) (1996)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Face/Off (1997)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Who\'s Afraid of Virginia Woolf? (1966)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Why Do Fools Fall In Love? (1998)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Police Academy 5: Assignment: Miami Beach (1988)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Communion (a.k.a. Alice, Sweet Alice/Holy Terror) (1977)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Brother, Can You Spare a Dime? (1975)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Who Framed Roger Rabbit? (1988)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='They Shoot Horses, Don\'t They? (1969)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Where\'s Marlowe? (1999)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Isn\'t She Great? (2000)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='I Am Cuba (Soy Cuba/Ya Kuba) (1964)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Beloved/Friend (Amigo/Amado) (1999)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='What Planet Are You From? (2000)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Who\'s Harry Crumb? (1989)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Who\'s That Girl? (1987)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Smoking/No Smoking (1993)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='What Ever Happened to Baby Jane? (1962)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='8 1/2 Women (1999)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Nine 1/2 Weeks (1986)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='F/X (1986)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='F/X 2 (1992)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='What About Bob? (1991)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Whatever Happened to Aunt Alice? (1969)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Naked Gun 2 1/2: The Smell of Fear, The (1991)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='X: The Unknown (1956)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Three Colors: Red (Trois couleurs: Rouge) (1994)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Three Colors: Blue (Trois couleurs: Bleu) (1993)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Three Colors: White (Trzy kolory: Bialy) (1994)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Highlander III: The Sorcerer (a.k.a. Highlander: The Final Dimension) (1994)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Supercop (Police Story 3: Supercop) (Jing cha gu shi III: Chao ji jing cha) (1992)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Halloween: The Curse of Michael Myers (Halloween 6: The Curse of Michael Myers) (1995)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Tie Me Up! Tie Me Down! (Â¡Ãtame!) (1990)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='8 1/2 (8Â½) (1963)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='First Strike (Police Story 4: First Strike) (Ging chaat goo si 4: Ji gaan daan yam mo) (1996)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Four Days in September (O Que Ã\x89 Isso, Companheiro?) (1997)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Freddy\'s Dead: The Final Nightmare (Nightmare on Elm Street Part 6: Freddy\'s Dead, A) (1991)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Halloween H20: 20 Years Later (Halloween 7: The Revenge of Laurie Strode) (1998)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Marcello Mastroianni: I Remember Yes, I Remember (Marcello Mastroianni: mi ricordo, sÃ¬, io mi ricordo) (1997)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Where\'s Marlowe? (1998)'].index, inplace=True)
movies.drop(movies.loc[movies['Title']=='Shall We Dance? (1996)'].index, inplace=True)

print(movies.head())

L=movies.Title
print(L)
moviestitles=list(L)
print(moviestitles)


for i in moviestitles:      
    titre=i+'.csv'
    # Open/create a file to append data to
    csvFile = open(titre, 'a')    
    #Use csv writer
    csvWriter = csv.writer(csvFile)
    client = tweepy.Client(bearer)   
    print(i)   
    begin_date=dt.date(2020,4,1)
    end_date=dt.date(2020,4,3)
    #Providing limit to the fetched tweets and also specifying the language of the desired tweets
    limit=1000
    language='english'
    x=[i] # ligne ajoutee au fichier :
    try:
        tweets = client.search_recent_tweets(query=i, tweet_fields=['author_id','context_annotations', 
                                                                    'created_at','conversation_id','geo',
                                                                    'possibly_sensitive','text'], max_results=100)
           
    except Exception:
        #creation fichier movies for which tweets were blacklisted      
        with open('BlackList.csv', 'a', newline='') as blackobject:              
            writer_object = writer(blackobject)            
            writer_object.writerow(x)             
            blackobject.close()
    else:
        #creation fichier movies for which tweets were extracted       
        with open('WhiteList.csv', 'a', newline='',encoding='utf-8') as whiteobject:              
            writer_object = writer(whiteobject)            
            writer_object.writerow(x)             
            whiteobject.close()
        
        #creation fichier movies for which tweets were extracted    
        if (tweets.data is not None ):
            #j=j+1
            for tweet in tweets.data:
                if len(tweet.context_annotations) > 0:
                    csvWriter.writerow([tweet.author_id,tweet.created_at, tweet.conversation_id, tweet.geo,
                                    tweet.possibly_sensitive, tweet.text.encode('utf-8') ]) 
            csvFile.close()  
            
            

