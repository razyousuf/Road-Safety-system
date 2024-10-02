"""
################
#    main.py   #
################
This FastAPI application integrates predictive and generative AI models to enhance road safety. 
It provides REST API endpoints to:
- Predict accident severity based on live data (e.g., traffic, weather, location).
- Generate contextual safety messages using generative AI.
- Visualize accident-prone areas through dynamic heatmaps.
- Convert safety messages into audio responses using Google Text-to-Speech.

Main features:
- Accident severity prediction using live and pre-existing data.
- Generating and delivering AI-powered safety messages in text and audio formats.
- Route-based accident heatmaps and visualization using folium.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import utils
from pydantic import BaseModel
from generative_ai import generate_safety_message
from gtts import gTTS
import folium
import csv
from fastapi import Depends
#import logging

class AccidentData(BaseModel):
    Date: str
    Time: str
    Day_of_Week: str
    Junction_Control: str
    Junction_Detail: str
    Light_Conditions: str
    Carriageway_Hazards: str
    Road_Surface_Conditions: str
    Road_Type: str
    Urban_or_Rural_Area: str
    Weather_Conditions: str
    Police_Force: str
    Vehicle_Type: str
    Local_Authority: str
    Latitude: float
    Longitude: float
    Number_of_Casualties: int
    Number_of_Vehicles: int
    Speed_limit: int

class GenerativeInput(BaseModel):
    latitude: float
    longitude: float

class RouteInput(BaseModel):
    start_location: str
    end_location: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    utils.load_saved_artifacts()  # Load models and necessary artifacts
    yield

# Create the FastAPI app and metadata
app = FastAPI(
    title="Road Safety Enhancement API",
    description="""
    This API integrates predictive and generative AI models to enhance road safety. 
    It uses historical accident data, Google Maps APIs, and generative AI models to predict accident severity, 
    generate safety alerts, and visualize accident-prone areas along specified routes.
    """,
    version="1.0.0",
    contact={
        "name": "Raz Yousufi",
        "url": "https://www.hope.ac.uk/mathematicsandcomputerscience/",
        "email": "razyousufi350@gmail.com",
    },
    lifespan=lifespan
)


# Utility function to process date and time
def process_date_time(features):
    day, month, year = utils.process_date(features['Date'])
    time_minutes = utils.process_time(features['Time'])
    return day, month, year, time_minutes


# Handle numerical and categorical input processing
def prepare_input_data(features):
    # Get columns from utils
    columns = utils.get_columns()
    # Initialize the input array with zeros
    input_data = np.zeros(len(columns))  
    
    # Process date and time
    day, month, year, time_minutes = process_date_time(features)
    # Create a dictionary to map column names to their indices
    column_indices = {col: idx for idx, col in enumerate(columns)}
    # Assign date and time values to the appropriate columns
    input_data[column_indices['Day']] = day
    input_data[column_indices['Month']] = month
    input_data[column_indices['Year']] = year
    input_data[column_indices['Time_Minutes']] = time_minutes

    # Handle categorical input processing using the utils function
    utils.handle_categorical_input(input_data, features)
    
    # Handle numerical input processing using the utils function
    utils.handle_numerical_input(input_data, features)

    # Wrap input_data in a pandas DataFrame with the correct column names
    input_data_df = pd.DataFrame([input_data], columns=columns)
    
    # Log the final input data for debugging
    #input_data_df.to_csv('static/input_data_log.csv', index=False)
    #print(f"Final Input: {input_data_df}")
    
    return input_data_df


@app.post('/Test_prediction')
async def test_prediction(data: AccidentData):
    try:
        # Convert the input data (received as JSON) into a dictionary for processing
        features = data.model_dump()
        
        # Prepare input data
        input_data_df = prepare_input_data(features)
        
        # Now pass the DataFrame to the model for prediction
         #prediction = utils.get_model().predict(input_data_df.to_numpy())
        prediction = utils.get_model().predict(input_data_df)
        result = int(prediction[0])

        return {'Accident_Severity': result}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in request data: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post('/predict-from-coordinates')
async def predict_from_coordinates(latitude: float, longitude: float):
    try:
        live_data = utils.fetch_live_data(latitude, longitude)  # Fetch live data
        features = utils.preprocess_live_data(live_data)  # Process live data
        
        # Prepare input data
        input_data_df = prepare_input_data(features)
        
        # Now pass the DataFrame to the model for prediction
        prediction = utils.get_model().predict(input_data_df)
        result = int(prediction[0])

        return {'Accident_Severity': result}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in request data: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post('/generative-response')
async def generative_response(latitude: float, longitude: float, severity: dict = Depends(predict_from_coordinates)):
    try:
        # Fetch live data and preprocess it
        live_data = utils.fetch_live_data(latitude, longitude)
        features = utils.preprocess_live_data(live_data)

        # Fetch traffic data and update features
        traffic_data = utils.fetch_here_traffic_data(latitude, longitude)
        features.update(traffic_data)
        features['current_traffic'] = traffic_data.get('Traffic_Condition', 'Unknown')

        # Use predicted severity
        features['predicted_severity'] = severity['Accident_Severity']

        # Generate safety message
        response_message = generate_safety_message(features)

        return {"generative_response": response_message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post('/generative-response-audio')
async def generative_response_audio(latitude: float, longitude: float, severity: dict = Depends(predict_from_coordinates)):
    try:
        # Get generative response
        generative_response_result = await generative_response(latitude, longitude, severity)
        response_message = generative_response_result["generative_response"]

        # Convert message to audio
        tts = gTTS(response_message, lang='en')
        audio_file_path = "safety_message.mp3"
        tts.save(audio_file_path)

        # Return the audio file as a response
        return FileResponse(audio_file_path, media_type='audio/mpeg', filename=audio_file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post('/route-accident-heatmap', response_class=HTMLResponse)
async def route_accident_heatmap(route_input: RouteInput):
    try:
        # Extract input data for route
        start_location = route_input.start_location
        end_location = route_input.end_location

        # Fetch route points and create a map
        route_points, polyline = utils.get_route(start_location, end_location)

        map_center = route_points[0]
        base_map = folium.Map(location=map_center, zoom_start=13)

        severity_colors = {0: 'red', 
                           1: 'orange', 
                           2: 'blue'}

        folium.PolyLine(route_points, color='green', weight=5).add_to(base_map)

        for lat, lon in route_points:
            live_data = utils.fetch_live_data(lat, lon)
            features = utils.preprocess_live_data(live_data)

            # Prepare input data
            input_data_df = prepare_input_data(features)

            # Make predictions
            prediction = utils.get_model().predict(input_data_df)
            severity = int(prediction[0])

            # Add marker for predicted severity
            folium.CircleMarker(
                location=(lat, lon),
                radius=0.3,
                color=severity_colors.get(severity, 'black'),
                fill=True,
                fill_color=severity_colors.get(severity, 'black'),
                fill_opacity=0.99
            ).add_to(base_map)

        heatmap_path = "static/route_heatmap.html"
        base_map.save(heatmap_path)

        with open(heatmap_path, 'r') as f:
            return HTMLResponse(content=f.read())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/accident-heatmap", response_class=HTMLResponse)
async def accident_heatmap():
    try:
        coordinates = load_coordinates_from_file()  # Load previously saved coordinates

        map_center = [51.5074, -0.1278]  # London center
        base_map = folium.Map(location=map_center, zoom_start=13)

        severity_colors = {0: 'red', 
                           1: 'orange', 
                           2: 'blue'}

        # Add accident markers to the map
        for lat, lon, severity in coordinates:
            folium.CircleMarker(
                location=(lat, lon),
                radius=0.2,
                color=severity_colors.get(severity, 'green'),
                fill=True,
                fill_color=severity_colors.get(severity, 'green'),
                fill_opacity=0.99
            ).add_to(base_map)

        heatmap_path = "static/heatmap.html"
        base_map.save(heatmap_path)

        with open(heatmap_path, 'r') as f:
            return HTMLResponse(content=f.read())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


#def save_coordinates_to_file(latitude, longitude, severity):
#    with open('static/coordinates.csv', 'a', newline='') as csvfile:
#        writer = csv.writer(csvfile)
#        writer.writerow([latitude, longitude, severity])

def load_coordinates_from_file():
    coordinates = []
    with open('static/coordinates.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            lat = float(row[0])
            lon = float(row[1])
            severity_code = int(row[2]) 
            coordinates.append([lat, lon, severity_code])
    return coordinates
