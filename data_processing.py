# This file contains the code for loading, cleaning, preprocessing, normalizing, validating, and splitting the data, using Pandas, Numpy, and Scikit-learn libraries.

# Import the necessary modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# Define the file names of the data
numerical_data_file = "numerical_data.csv"
textual_data_file = "textual_data.json"
image_data_file = "image_data.csv"
audio_data_file = "audio_data.json"

# Define the empty dataframes for the data
numerical_data = pd.DataFrame()
textual_data = pd.DataFrame()
image_data = pd.DataFrame()
audio_data = pd.DataFrame()

# Load the numerical data from the CSV file, using Pandas library
numerical_data = pd.read_csv(numerical_data_file)
# Load the textual data from the JSON file, using Pandas library
textual_data = pd.read_json(textual_data_file, orient="records")
# Load the image data from the CSV file, using Pandas library
image_data = pd.read_csv(image_data_file)
# Load the audio data from the JSON file, using Pandas library
audio_data = pd.read_json(audio_data_file, orient="records")

# Clean and preprocess the numerical data, such as removing outliers, null values, duplicates, or noise, using Pandas and Numpy libraries
numerical_data = numerical_data.dropna() # Drop the rows with null values
numerical_data = numerical_data.drop_duplicates() # Drop the rows with duplicate values
numerical_data = numerical_data[numerical_data["Volume"] > 0] # Drop the rows with zero volume
numerical_data = numerical_data[(np.abs(numerical_data - numerical_data.mean()) <= 3 * numerical_data.std()).all(axis=1)] # Drop the rows with outliers

# Clean and preprocess the textual data, such as removing punctuation, stopwords, numbers, or links, using Pandas and Numpy libraries
textual_data = textual_data.dropna() # Drop the rows with null values
textual_data = textual_data.drop_duplicates() # Drop the rows with duplicate values
textual_data["Text"] = textual_data["Text"].str.lower() # Convert the text to lowercase
textual_data["Text"] = textual_data["Text"].str.replace("[^\w\s]", "") # Remove the punctuation
textual_data["Text"] = textual_data["Text"].str.replace("\d+", "") # Remove the numbers
textual_data["Text"] = textual_data["Text"].str.replace("http\S+", "") # Remove the links
textual_data["Text"] = textual_data["Text"].apply(lambda x: " ".join([word for word in x.split() if word not in stopwords.words("english")])) # Remove the stopwords

# Clean and preprocess the image data, such as removing null values, duplicates, or invalid URLs, using Pandas and Numpy libraries
image_data = image_data.dropna() # Drop the rows with null values
image_data = image_data.drop_duplicates() # Drop the rows with duplicate values
image_data = image_data[image_data["Image_URL"].str.startswith("https")] # Drop the rows with invalid URLs

# Clean and preprocess the audio data, such as removing null values, duplicates, or unintelligible speech, using Pandas and Numpy libraries
audio_data = audio_data.dropna() # Drop the rows with null values
audio_data = audio_data.drop_duplicates() # Drop the rows with duplicate values
audio_data = audio_data[audio_data["Speech"] != "Could not understand audio"] # Drop the rows with unintelligible speech

# Normalize and standardize the numerical data, such as scaling, transforming, or encoding the data, using Scikit-learn library
scaler = MinMaxScaler() # Create a scaler object
numerical_data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = scaler.fit_transform(numerical_data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]) # Scale the numerical columns
encoder = OneHotEncoder() # Create an encoder object
numerical_data["Symbol"] = encoder.fit_transform(numerical_data[["Symbol"]]).toarray() # Encode the symbol column

# Normalize and standardize the textual data, such as transforming or encoding the data, using Scikit-learn library
encoder = LabelEncoder() # Create an encoder object
textual_data["Sentiment"] = encoder.fit_transform(textual_data["Sentiment"]) # Encode the sentiment column
textual_data["Keyword"] = encoder.fit_transform(textual_data["Keyword"]) # Encode the keyword column

# Normalize and standardize the image data, such as encoding the data, using Scikit-learn library
encoder = OneHotEncoder() # Create an encoder object
image_data["Symbol"] = encoder.fit_transform(image_data[["Symbol"]]).toarray() # Encode the symbol column

# Normalize and standardize the audio data, such as encoding the data, using Scikit-learn library
encoder = LabelEncoder() # Create an encoder object
audio_data["Keyword"] = encoder.fit_transform(audio_data["Keyword"]) # Encode the keyword column

# Validate and split the numerical data, such as checking the validity, reliability, and distribution of the data, and dividing the data into training, validation, and testing sets, using Scikit-learn library
numerical_data.info() # Print the information of the numerical data
numerical_data.describe() # Print the summary statistics of the numerical data
numerical_data.hist() # Plot the histograms of the numerical data
numerical_data_train, numerical_data_test = train_test_split(numerical_data, test_size=0.2, random_state=42) # Split the numerical data into training and testing sets
numerical_data_train, numerical_data_val = train_test_split(numerical_data_train, test_size=0.25, random_state=42) # Split the numerical data training set into training and validation sets

# Validate and split the textual data, such as checking the validity, reliability, and distribution of the data, and dividing the data into training, validation, and testing sets, using Scikit-learn library
textual_data.info() # Print the information of the textual data
textual_data.describe() # Print the summary statistics of the textual data
textual_data["Sentiment"].value_counts().plot(kind="bar") # Plot the bar chart of the sentiment counts
textual_data["Keyword"].value_counts().plot(kind="bar") # Plot the bar chart of the keyword counts
textual_data_train, textual_data_test = train_test_split(textual_data, test_size=0.2, random_state=42) # Split the textual data into training and testing sets
textual_data_train, textual_data_val = train_test_split(textual_data_train, test_size=0.25, random_state=42) # Split the textual data training set into training and validation sets

# Validate and split the image data, such as checking the validity, reliability, and distribution of the data, and dividing the data into training, validation, and testing sets, using Scikit-learn library
image_data.info() # Print the information of the image data
image_data.describe() # Print the summary statistics of the image data
image_data["Symbol"].value_counts().plot(kind="bar") # Plot the bar chart of the symbol counts
image_data_train, image_data_test = train_test_split(image_data, test_size=0.2, random_state=42) # Split the image data into training and testing sets
image_data_train, image_data_val = train_test_split(image_data_train, test_size=0.25, random_state=42) # Split the image data training set into training and validation sets

# Validate and split the audio data, such as checking the validity, reliability, and distribution of the data, and dividing the data into training, validation, and testing sets, using Scikit-learn library
audio_data.info() # Print the information of the audio data
audio_data.describe() # Print the summary statistics of the audio data
audio_data["Keyword"].value_counts().plot(kind="bar") # Plot the bar chart of the keyword counts
audio_data_train, audio_data_test = train_test_split(audio_data, test_size=0.2, random_state=42) # Split the audio data into training and testing sets
audio_data_train, audio_data_val = train_test_split(audio_data_train, test_size=0.25, random_state=42) # Split the audio data training set into training and validation sets
