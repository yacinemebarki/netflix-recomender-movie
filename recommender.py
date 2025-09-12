import pandas as pd
import numpy as np  
import pickle
with open("user_category.pkl", "rb") as f:
    user_category = pickle.load(f)
with open("my_model.pkl","rb") as f:
    model=pickle.load(f)
    
movie=pd.read_csv("movies.csv")
movie=movie[["title","content_type","genre_primary","release_year","duration_minutes","country_of_origin","language","content_warning"]]
movie["content_warning"]=movie["content_warning"].replace({
    True:1,
    False:0
})
def bot(movie):
    age=int(input("enter your age:\n"))
    found=False
    subscription=[0,1,2,3]
    while found==False:
        subscription_plans=int(input("if you use basic enter 0 if you user standard enter 1 if you use premium enter 2 if you use premium+ enter 3:\n"))
        if subscription_plans in subscription:
            found=True
    gender_arr=[1,0]
    found=False
    while found==False:
        gender=int(input("if you are male enter 1 if you are female enter 0:\n"))
        if gender in gender_arr:
            found=True
    found=False
    while found==False:
        country=int(input("if you are from USA enter 1 if you are from CANADA enter 0:\n"))
        if country in gender_arr:
            found=True
    user=user_category.predict([[age,gender,country,subscription_plans]])
    print(user)
    max_prob=0
    movie_probs=[]
    for i in range(len(movie)):
        x = movie.iloc[i]
        user_cat = int(user[0])
        x_test = pd.DataFrame([{
            "user_category": user_cat,
            "content_type": x["content_type"],
    
            "genre_primary": x["genre_primary"],
            "release_year": x["release_year"],
            "duration_minutes": x["duration_minutes"],
            "country_of_origin": x["country_of_origin"],
            "language": x["language"],
            "content_warning": x["content_warning"]
        }])
        prob = model.predict_proba(x_test)
        current_prob = prob[0][2]
        movie_probs.append((x["title"], current_prob))

movie_probs.sort(key=lambda x: x[1], reverse=True)
      
    print(movie_probs)
    print("enjoy your movie")
bot(movie)                     





        
