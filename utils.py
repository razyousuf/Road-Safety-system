"""
#################
#    utils.py   #
#################
This module provides utility functions to handle live traffic, weather, and geolocation data. It includes methods for:
- Loading saved machine learning artifacts and encoders.
- Fetching live data from HERE, Google, and OpenWeather APIs.
- Processing and preprocessing the live data for accident severity prediction.
- Handling categorical feature encoding.
- Route fetching and polyline decoding using Google Maps.

Main features:
- Fetching traffic data and junction details from the HERE API.
- Handling geolocation via Google Geocode API.
- Integrating real-time weather data using the OpenWeather API.
"""

import joblib
import json
import numpy as np
from datetime import datetime
import logging
import requests
import polyline
import csv
import os
import math


model = None
label_encoders = None
top_k_categories = None
columns = None
urban_areas = []
road_data = {}

# Configure logging to display INFO and WARNING messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Keys
GOOGLE_API_KEY = 'Your_Google_Cloud_API_Key'
WEATHER_API_KEY = 'Your_Weather_API_KEY'
HERE_API_KEY = 'Your_HERE_API_KEY'

# Global variables
road_data = {}

def load_saved_artifacts():
    global model, label_encoders, top_k_categories, columns, urban_areas, road_data

    model = joblib.load("artifacts/model.joblib")
    if not hasattr(model, 'predict'):
        raise ValueError("Loaded model is not a valid scikit-learn estimator")

    label_encoders = joblib.load("artifacts/category_label_encoders.joblib")

    with open("artifacts/top_k_categories.json", "r") as f:
        top_k_categories = json.load(f)

    with open("artifacts/columns.json", "r") as f:
        columns = json.load(f)["data_columns"]

    with open("artifacts/urban.txt", "r") as f:
        urban_areas = [line.strip() for line in f if line.strip()]

    # Load road data (latitude and longitude keys) without Speed_limit
    #road_data = {}  # Initialize the road data dictionary
    with open("static/road_data.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Ensure the values are floats and add them as keys in the road_data dictionary
                key = (float(row["Latitude"]), float(row["Longitude"]))
                road_data[key] = {
                    "Junction_Control": row["Junction_Control"],
                    "Junction_Detail": row["Junction_Detail"],
                    "Road_Type": row["Road_Type"]
                }
            except ValueError as e:
                logging.warning(f"Skipping invalid latitude/longitude: {row['Latitude']}, {row['Longitude']} - Error: {e}")

    logging.info(f"Artifacts and road data loaded successfully")


def get_model():
    return model
def get_columns():
    return columns

# Procss Date
def process_date(date_str: str):
    try:
        date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        return date_obj.day, date_obj.month, date_obj.year
    except ValueError as e:
        raise ValueError(f"Date format error: {str(e)}")


# Procss Time
def process_time(time_str: str):
    time = datetime.strptime(time_str, "%H:%M")
    return time.hour * 60 + time.minute

# Handle Categorical data
def handle_categorical_input(input_data, data):
    for feature, top_categories in top_k_categories.items():
        #logging.info(f"Feature is: {feature}. Top Categories are: {top_categories}.")

        if feature in data:
            value = data[feature]

            # Handle "Unknown" values for Local_Authority, Police_Force, and Vehicle_Type
            if feature == "Local_Authority" and value not in top_k_categories["Local_Authority"]:
                value = "Other"
            elif feature == "Police_Force" and value not in top_k_categories["Police_Force"]:
                value = "Other"
            elif feature == "Vehicle_Type" and value not in top_k_categories["Vehicle_Type"]:
                value = "Other"

            # Mapping feature to the column format
            normalized_feature = feature.replace(" ", "_")
            column_name = f"{normalized_feature}_{value}"

            # Log the feature and value
            #logging.info(f"Original feature: {feature}, value: {value}")
            #logging.info(f"Formatted as: {column_name}")

            if column_name in columns:
                #logging.info(f"Column found: {column_name}. Index: {columns.index(column_name)}")
                input_data[columns.index(column_name)] = 1
            else:
                logging.warning(f"Column {column_name} not found. Continuing with fallback 'Other'.")

                # Handle fallback to 'Other' if not found in columns
                generalized_column_name = f"{normalized_feature}_Other"
                if generalized_column_name in columns:
                    input_data[columns.index(generalized_column_name)] = 1
                else:
                    logging.warning(f"No generalized column found for {feature}. Continuing with zeros.")

     # (2) Apply label encoding for features that require it
    cat_cols_labling = ['Day_of_Week', 'Junction_Control', 'Junction_Detail', 'Light_Conditions',
                        'Carriageway_Hazards', 'Road_Surface_Conditions', 'Road_Type', 'Urban_or_Rural_Area',
                        'Weather_Conditions']

    for feature in cat_cols_labling:
        if feature in data:
            value = data[feature]
            # Log the raw feature and value
            #logging.info(f"Feature: {feature}, Raw value: {value}")
            if feature in label_encoders:
                try:
                    # Transform the feature value using the stored label encoder
                    encoded_value = label_encoders[feature].transform([value])[0]
                    # Assign the encoded value to the appropriate column in input_data
                    input_data[columns.index(feature)] = encoded_value
                    # Log the encoded value
                    #logging.info(f"Encoded {feature}: {encoded_value} (Original value: {value})")
                except ValueError:
                    logging.warning(f"Value '{value}' not found in the label encoder for {feature}. Using default value.")
                    # Handle unknown values, assign a default value if label not found
                    input_data[columns.index(feature)] = 0  # Default to 0 for unknown categories

    logging.info(f"Categorical input: {input_data}")

# Handle Numerical Data
def handle_numerical_input(input_data, data):
    # List of numerical features that you expect in your dataset
    numerical_features = ['Latitude', 'Longitude', 'Speed_limit', 'Number_of_Casualties', 'Number_of_Vehicles']

    for feature in numerical_features:
        if feature in data:
            value = data[feature]
            try:
                # Ensure the value is converted to float or int as per your model training
                if feature in ['Latitude', 'Longitude']:
                    input_data[columns.index(feature)] = float(value)
                elif feature in ['Speed_limit', 'Number_of_Casualties', 'Number_of_Vehicles']:
                    input_data[columns.index(feature)] = int(value)
            except ValueError as e:
                logging.warning(f"Invalid value for {feature}: {value}. Defaulting to 0.")
                input_data[columns.index(feature)] = 0

    logging.info(f"Numerical and Categorical input: {input_data}")

# ________________________________________________________________________________________________________________

def fetch_here_traffic_data(latitude, longitude):
    # HERE Traffic API to get traffic-related information such as road types, junctions, and hazards
    incident_url = "https://data.traffic.hereapi.com/v7/incidents"
    flow_url = "https://data.traffic.hereapi.com/v7/flow"

    # Define the geographical area as a circle around the provided latitude and longitude
    location_area = f"circle:{latitude},{longitude};r=500"  # 500 meters radius

    # Request parameters for incidents
    incident_params = {
        'apiKey': HERE_API_KEY,
        'in': location_area,
        'criticality': 'minor,major,critical',  # Include all possible criticalities
        'locationReferencing': 'shape'
    }

    # Request parameters for traffic flow
    flow_params = {
        'apiKey': HERE_API_KEY,
        'in': location_area,
        'locationReferencing': 'shape'
    }

    # Fetch traffic incidents
    incident_response = requests.get(incident_url, params=incident_params)
    if incident_response.status_code != 200:
        logging.error(f"Incident API call failed: {incident_response.status_code} - {incident_response.text}")
        incidents = []
    else:
        incidents = incident_response.json().get('results', [])
        logging.info(f"Incidents data fetched from HERE API: {len(incidents)}")

    # Fetch traffic flow
    flow_response = requests.get(flow_url, params=flow_params)
    if flow_response.status_code != 200:
        logging.error(f"Flow API call failed: {flow_response.status_code} - {flow_response.text}")
        flows = []
    else:
        flows = flow_response.json().get('results', [])
        logging.info(f"Flow data fetched from HERE API: {len(flows)}")

    # Initialize default values
    #carriageway_hazards = 'Other object on road'
    speed_limit = int(0)  # Initialize to 0 to track the highest speed
    road_closure = False    
    max_jam_factor = 0  

    # Predefined list of known hazards
    known_hazards = [
        "Other object on road",
        "Any animal in carriageway (except ridden horse)",
        "Pedestrian in carriageway - not injured",
        "Previous accident",
        "Vehicle load on road"]
    # (1) Process incidents to extract hazards
    carriageway_hazards = "Unknown"
    for incident in incidents:
        # Check if 'incidentDetails' exists and extract 'type'
        incident_details = incident.get('incidentDetails', {})
        carriageway_hazards = incident_details.get('type', 'Unknown')  # Save 'type' in 'carriageway_hazards'
        
        # Check if the hazard is in the known hazards list
        if carriageway_hazards not in known_hazards:
            carriageway_hazards = "Other object on road"
        break  # Exit after processing the first incident

    # (2) Process incidents to get Road condition (Open or Closed)
    for closure in incidents:
        Closed_road = "Open"
        closure_details = closure.get('closureDetails', {})
        road_closure = closure_details.get('roadClosed', False) 
        if road_closure == True:
            Closed_road = "Closed"
        break  # Exit after processing the first road closure

    # (3) Process traffic flow to find the speed limit and jam factor
    for flow in flows:
        current_speed = flow.get('currentFlow', {}).get('speed', 0)
        jam_factor = flow.get('currentFlow', {}).get('jamFactor', 0.0)
        if current_speed > speed_limit:
            speed_limit = current_speed
        if jam_factor > max_jam_factor:
            max_jam_factor = jam_factor

    # Determine traffic condition based on the max jam factor (Traffic Condition)
    traffic_condition = "Normal"
    if max_jam_factor > 4.0 and max_jam_factor <= 7.0:
        traffic_condition = "Heavy"
    elif max_jam_factor > 7.0:
        traffic_condition = "Severe"

    # Determine speed limit according to the limits during the model training
    # Available speed limits from the training data
    training_speed_limits = [30, 20, 50, 40, 70, 60, 10, 15]
    
    # Convert the speed limit to the nearest value from the training data
    if speed_limit > 0:  # Only process if there's a valid speed
        speed_limit = min(training_speed_limits, key=lambda x: abs(x - speed_limit))
    # If no speed data was found, return 'Unknown'
    # Convert the final speed_limit to an integer
    speed_limit = int(speed_limit)
    speed_limit = speed_limit if speed_limit > 0 else "Unknown"

    # Return both traffic data and traffic condition in the same dictionary
    return {
        'Carriageway_Hazards': carriageway_hazards,
        'Speed_limit': speed_limit,
        'Traffic_Condition': traffic_condition
    }

# ________________________________________________________________________________________________________________

def fetch_live_data(latitude, longitude):
    # Google Geocode API to get location information
    geocode_api_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={latitude},{longitude}&key={GOOGLE_API_KEY}"
    # OpenWeather API to get weather information
    weather_api_url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={WEATHER_API_KEY}&units=metric"

    geocode_response = requests.get(geocode_api_url)
    weather_response = requests.get(weather_api_url)
    
    # Log the response for debugging
    if geocode_response.status_code == 200:
        logging.info(f"Geocode data fetched from Google API: {len(geocode_response.json())}")
        geocode_data = geocode_response.json()
    else:
         logging.warning(f"Error fetching Google Geocode data. Status code: {geocode_response.status_code}")
    
    if weather_response.status_code == 200:
        logging.info(f"Weather data fetched from OpenWeatherMap API: {len(weather_response.json())}")
        weather_data = weather_response.json()
    else:
        logging.warning(f"Error fetching Weather data. Status code: {weather_response.status_code}")

    # Fetch traffic data from HERE
    traffic_data = fetch_here_traffic_data(latitude, longitude)
    # Log traffic data
    logging.info(f"Traffic data: {traffic_data}")

    # Find nearest coordinates in road_data
    nearest_road_key, min_distance = find_nearest_coordinates(latitude, longitude, road_data)
    #road_key = (round(latitude, 5), round(longitude, 5))
    #nearest_road_key, min_distance = find_nearest_coordinates(round(latitude, 5), round(longitude, 5), road_data)
    # Find the nearest road data (Junction_Control, Junction_Detail, Road_Type)
    #nearest_road_key = find_nearest_coordinates(latitude, longitude, road_data)
    road_info = road_data.get(nearest_road_key, {
        'Junction_Control': 'Data missing or out of range', #'Unknown',
        'Junction_Detail': 'Other junction',
        'Road_Type': 'Unknown'
    })
    # Get road data (Junction_Control, Junction_Detail, Road_Type)
    #road_key = (latitude, longitude)
    #road_info = road_data.get(road_key, {
    #    'Junction_Control': 'Unknown',
    #    'Junction_Detail': 'Unknown',
    #    'Road_Type': 'Unknown'
    #})

    # Combine traffic data with Google geocode and OpenWeather data
    return {
        'geocode': geocode_data,
        'weather': weather_data,
        'traffic': traffic_data,
        'road': road_info
    }

# ________________________________________________________________________________________________________________


def preprocess_live_data(live_data):
    try:
        logging.debug("Processing live data")

        # Check if geocode, weather, and traffic data exist
        if 'geocode' not in live_data or 'weather' not in live_data or 'traffic' not in live_data or 'road' not in live_data:
            logging.error("Geocode, weather, or traffic data is missing from live data")
            raise ValueError("Geocode, weather, or traffic data missing")

        geocode_results = live_data['geocode'].get('results', [])
        weather_data = live_data['weather']
        traffic_data = live_data['traffic']
        road_data = live_data['road']

        if not geocode_results:
            logging.error("No results found in geocode data")
            raise ValueError("No results found in geocode data")

        # (1) Infer the first three components(Urban|Rural, Local Authority and Police Force) from the fetched live data
        all_postal_towns = []
        all_local_authorities = []
        all_police_forces = []

        for result in geocode_results:
            address_components = result.get('address_components', [])
            for component in address_components:
                types = component['types']

                # Collect 'postal_town' values
                if 'postal_town' in types:
                    all_postal_towns.append(component['long_name'])
                    all_local_authorities.append(component['long_name'])

                # Collect 'administrative_area_level_2' values (for both Local Authority and Police Force)
                if 'administrative_area_level_2' in types:
                    all_police_forces.append(component['long_name'])

        # Compare all collected 'postal_town' values against top_k_categories['Local_Authority']
        postal_city = None
        local_authority = None
        for town in all_postal_towns:
            if town in top_k_categories['Local_Authority']:
                postal_city = town
                local_authority = town
                break
        if not postal_city:
            postal_city = all_postal_towns[0] if all_postal_towns else 'Other'
        if not local_authority:
            local_authority = all_local_authorities[0] if all_local_authorities else 'Other'

        # Compare all collected 'administrative_area_level_2' values against top_k_categories['Police_Force']
        police_force = None
        for force in all_police_forces:
            if force in top_k_categories['Police_Force']:
                police_force = force
                break
        if not police_force:
            police_force = all_police_forces[0] if all_police_forces else 'Other'

        # Urban or Rural classification based on postal_city
        area_type = 'Urban' if postal_city in urban_areas else 'Rural'

        # (2) Infer 'Light_Conditions' based on current time and sunrise/sunset
        current_timestamp = datetime.now().timestamp()
        light_conditions = 'Daylight' if current_timestamp < weather_data['sys']['sunset'] and current_timestamp > weather_data['sys']['sunrise'] else 'Darkness - no lighting'


        # (3) Infer 'Weather_Conditions' based on weather description
        weather_mapping = {
            "Fine": "Fine no high winds",
            "Raining": "Raining no high winds",
            "Snowing": "Snowing no high winds",
            "Fog": "Fog or mist",
            "Raining + high winds": "Raining + high winds",
            "Fine + high winds": "Fine + high winds",
            "Snowing + high winds": "Snowing + high winds"
        }
        weather_conditions = weather_data['weather'][0]['main']
        # Match the weather condition with the predefined mapping
        matched_weather = None
        for key in weather_mapping:
            if weather_conditions.startswith(key):
                matched_weather = weather_mapping[key]
                break
        # If no match is found, default to "Other"
        if matched_weather is None:
            matched_weather = "Other"

        # (4) Infer 'Road_Surface_Conditions' based on weather description
        road_surface_conditions = 'Wet or damp' if 'rain' in weather_conditions.lower() else 'Dry'

        # (5) Vehicle Type: Default to 'Car'
        vehicle_type = 'Car'
        if vehicle_type not in top_k_categories['Vehicle_Type']:
            vehicle_type = 'Other'

        # (6) Handling numerical features and filling them with defaults
        latitude = float(live_data['geocode']['results'][0].get('geometry', {}).get('location', {}).get('lat', 0))
        longitude = float(live_data['geocode']['results'][0].get('geometry', {}).get('location', {}).get('lng', 0))
        number_of_casualties = np.int64(4)  # Placeholder value, adjust as needed
        number_of_vehicles = np.int64(2)  # Placeholder value, adjust as needed

        # Prepare features dictionary
        features = {
            'Date': datetime.now().strftime("%d/%m/%Y"),
            'Time': datetime.now().strftime("%H:%M"),
            'Day_of_Week': datetime.now().strftime("%A"),
            'Junction_Control': road_data['Junction_Control'],
            'Junction_Detail': road_data['Junction_Detail'],
            'Light_Conditions': light_conditions,
            'Carriageway_Hazards': traffic_data['Carriageway_Hazards'],
            'Road_Surface_Conditions': road_surface_conditions,
            'Road_Type': road_data['Road_Type'],
            'Urban_or_Rural_Area': area_type,
            'Weather_Conditions': matched_weather,#weather_conditions,
            'Vehicle_Type': vehicle_type,
            'Police_Force': police_force,
            'Local_Authority': local_authority,
            'Speed_limit': traffic_data['Speed_limit'],
            'Latitude': latitude,
            'Longitude': longitude,
            'Number_of_Casualties': number_of_casualties,
            'Number_of_Vehicles': number_of_vehicles
        }

        logging.info(f"Processed features: {features}")
        return features

    except KeyError as e:
        logging.error(f"Missing key in live data: {str(e)}")
        raise ValueError(f"Missing key in live data: {str(e)}")
    except IndexError as e:
        logging.error(f"Unexpected structure in live data: {str(e)}")
        raise ValueError(f"Unexpected structure in live data: {str(e)}")
    except Exception as e:
        logging.error(f"Error processing live data: {str(e)}")
        raise ValueError(f"Error processing live data. Reason: {str(e)}")


# ________________________________________________________________________________________________________________

def get_route(start_location, end_location):
    directions_api_url = f"https://maps.googleapis.com/maps/api/directions/json?origin={start_location.replace(' ', '+')}&destination={end_location.replace(' ', '+')}&key={GOOGLE_API_KEY}"
    response = requests.get(directions_api_url)
    if response.status_code == 200:
        logging.info(f"Route data fetched from Google API: {len(response.json())}")
        directions_data = response.json()
        if directions_data['status'] == 'OK':
            route = directions_data['routes'][0]['overview_polyline']['points']
            route_points = polyline.decode(route)  # Decode the polyline to get route points
            return route_points, route  # Return route points and polyline
        else:
            raise ValueError(f"Error fetching route: {directions_data['status']}")
    else:
        raise ValueError(f"Error fetching route: {response.status_code}")



# Function to compute Haversine distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in kilometers

# Function to find nearest coordinates from road_data
def find_nearest_coordinates(latitude, longitude, road_data):
    closest_key = None
    min_distance = float('inf')

    for road_lat, road_lon in road_data.keys():
        distance = haversine_distance(latitude, longitude, road_lat, road_lon)
        if distance < min_distance:
            min_distance = distance
            closest_key = (road_lat, road_lon)

    return closest_key, min_distance
