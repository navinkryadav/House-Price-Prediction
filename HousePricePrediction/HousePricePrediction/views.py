from django.shortcuts import render
from joblib import load
import numpy as np

def index (request) :
    return render(request, "index.html")

def predict(request) :
    longitude = request.POST['longitude']
    latitude = request.POST['latitude']
    housing_median_age = request.POST['housing_median_age']
    total_rooms = request.POST['total_rooms']
    total_bedrooms = request.POST['total_bedrooms']
    population = request.POST['population']
    households = request.POST['households']
    median_income = request.POST['median_income']
    rooms_per_household  = request.POST['rooms_per_household']
    population_per_household = request.POST['population_per_household']
    bedrooms_per_room = request.POST['bedrooms_per_room']
    h_ocean = request.POST['1h_ocean']
    inland = request.POST['inland']
    island= request.POST['island']
    near_bay = request.POST['near_bay']
    near_ocean    = request.POST['near_ocean']
    

    feature = np.array([[longitude,latitude,housing_median_age,total_rooms,
                        total_bedrooms,population,households,median_income,
                        rooms_per_household,population_per_household,bedrooms_per_room,
                        h_ocean,inland,island,near_bay,near_ocean]], dtype="float64")


    model = load('D:/housing.joblib')
    prediction = model.predict(feature)
    result = {"prediction":prediction,"flag":True}
    return render(request,"index.html",result)