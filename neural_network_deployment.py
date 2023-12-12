# This file contains the code for deploying the natural language processing, generative adversarial, convolutional, recurrent, and attentional neural networks, using PyTorch and Flask libraries.

# Import the necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from flask import Flask, request, jsonify, render_template

# Define the hyperparameters of the neural networks
batch_size = 32 # The number of samples in a batch
num_epochs = 10 # The number of epochs to train the neural networks
learning_rate = 0.001 # The learning rate of the optimizer
hidden_size = 256 # The size of the hidden layer
num_classes = 3 # The number of classes for the sentiment and contextual labels
num_features = 5 # The number of features for the symbol column
image_size = 64 # The size of the image data
num_channels = 3 # The number of channels of the image data
latent_size = 100 # The size of the latent vector for the generative adversarial network
num_heads = 4 # The number of heads for the multi-head attention
dropout_rate = 0.2 # The dropout rate for the neural networks

# Define the device for the neural networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the custom dataset classes for the data
class NumericalDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.x = torch.tensor(data[["Open", "High", "Low", "Close", "Adj Close", "Volume"] + [f"Symbol_{i}" for i in range(num_features)]].values, dtype=torch.float)
        self.y = torch.tensor(data[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].values, dtype=torch.float)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class TextualDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.x = data["Text"].values
        self.y = torch.tensor(data["Sentiment"].values, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class ImageDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.x = torch.tensor(data["Image_URL"].apply(lambda x: requests.get(x).content).apply(lambda x: Image.open(io.BytesIO(x))).apply(lambda x: x.resize((image_size, image_size))).apply(lambda x: np.array(x)).values.tolist(), dtype=torch.float).permute(0, 3, 1, 2) / 255.0
        self.y = torch.tensor(data["Symbol"].values, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.x = data["Speech"].values
        self.y = torch.tensor(data["Keyword"].values, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]

# Load the data from the files, using Pandas library
numerical_data_train = pd.read_csv("numerical_data_train.csv")
numerical_data_val = pd.read_csv("numerical_data_val.csv")
numerical_data_test = pd.read_csv("numerical_data_test.csv")
textual_data_train = pd.read_json("textual_data_train.json", orient="records")
textual_data_val = pd.read_json("textual_data_val.json", orient="records")
textual_data_test = pd.read_json("textual_data_test.json", orient="records")
image_data_train = pd.read_csv("image_data_train.csv")
image_data_val = pd.read_csv("image_data_val.csv")
image_data_test = pd.read_csv("image_data_test.csv")
audio_data_train = pd.read_json("audio_data_train.json", orient="records")
audio_data_val = pd.read_json("audio_data_val.json", orient="records")
audio_data_test = pd.read_json("audio_data_test.json", orient="records")

# Create the dataset objects for the data, using the custom dataset classes
numerical_dataset_train = NumericalDataset(numerical_data_train)
numerical_dataset_val = NumericalDataset(numerical_data_val)
numerical_dataset_test = NumericalDataset(numerical_data_test)
textual_dataset_train = TextualDataset(textual_data_train)
textual_dataset_val = TextualDataset(textual_data_val)
textual_dataset_test = TextualDataset(textual_data_test)
image_dataset_train = ImageDataset(image_data_train)
image_dataset_val = ImageDataset(image_data_val)
image_dataset_test = ImageDataset(image_data_test)
audio_dataset_train = AudioDataset(audio_data_train)
audio_dataset_val = AudioDataset(audio_data_val)
audio_dataset_test = AudioDataset(audio_data_test)

# Create the data loader objects for the data, using the dataset objects and the batch size
numerical_data_loader_train = DataLoader(numerical_dataset_train, batch_size=batch_size, shuffle=True)
numerical_data_loader_val = DataLoader(numerical_dataset_val, batch_size=batch_size, shuffle=False)
numerical_data_loader_test = DataLoader(numerical_dataset_test, batch_size=batch_size, shuffle=False)
textual_data_loader_train = DataLoader(textual_dataset_train, batch_size=batch_size, shuffle=True)
textual_data_loader_val = DataLoader(textual_dataset_val, batch_size=batch_size, shuffle=False)
textual_data_loader_test = DataLoader(textual_dataset_test, batch_size=batch_size, shuffle=False)
image_data_loader_train = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True)
image_data_loader_val = DataLoader(image_dataset_val, batch_size=batch_size, shuffle=False)
image_data_loader_test = DataLoader(image_dataset_test, batch_size=batch_size, shuffle=False)
audio_data_loader_train = DataLoader(audio_dataset_train, batch_size=batch_size, shuffle=True)
audio_data_loader_val = DataLoader(audio_dataset_val, batch_size=batch_size, shuffle=False)
audio_data_loader_test = DataLoader(audio_dataset_test, batch_size=batch_size, shuffle=False)

# Import the neural network models from the neural_network_design.py file
from neural_network_design import NLPNet, Generator, Discriminator, CNNNet, RNNNet, ATTNet

# Create the neural network objects for the models, using the device
nlp_net = NLPNet().to(device)
generator = Generator().to(device)
discriminator = Discriminator().to(device)
cnn_net = CNNNet().to(device)
rnn_net = RNNNet().to(device)
att_net = ATTNet().to(device)

# Define the loss functions for the models, using PyTorch library
nlp_loss = nn.CrossEntropyLoss() # Cross entropy loss for the natural language processing model
gan_loss = nn.BCELoss() # Binary cross entropy loss for the generative adversarial model
cnn_loss = nn.CrossEntropyLoss() # Cross entropy loss for the convolutional model
rnn_loss = nn.MSELoss() # Mean squared error loss for the recurrent model
att_loss = nn.CrossEntropyLoss() # Cross entropy loss for the attentional model

# Define the optimizers for the models, using PyTorch library
nlp_optimizer = optim.Adam(nlp_net.parameters(), lr=learning_rate) # Adam optimizer for the natural language processing model
generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate) # Adam optimizer for the generator model
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate) # Adam optimizer for the discriminator model
cnn_optimizer = optim.Adam(cnn_net.parameters(), lr=learning_rate) # Adam optimizer for the convolutional model
rnn_optimizer = optim.Adam(rnn_net.parameters(), lr=learning_rate) # Adam optimizer for the recurrent model
att_optimizer = optim.Adam(att_net.parameters(), lr=learning_rate) # Adam optimizer for the attentional model

# Import the fusion network model from the neural_network_fusion.py file
from neural_network_fusion import FusionNet

# Create the fusion network object for the model, using the device
fusion_net = FusionNet().to(device)

# Define the loss function for the fusion model, using PyTorch library
fusion_loss = nn.CrossEntropyLoss() # Cross entropy loss for the fusion model

# Define the optimizer for the fusion model, using PyTorch library
fusion_optimizer = optim.Adam(fusion_net.parameters(), lr=learning_rate) # Adam optimizer for the fusion model

# Load the trained models from the files, using PyTorch library
nlp_net.load_state_dict(torch.load("nlp_net.pth"))
generator.load_state_dict(torch.load("generator.pth"))
discriminator.load_state_dict(torch.load("discriminator.pth"))
cnn_net.load_state_dict(torch.load("cnn_net.pth"))
rnn_net.load_state_dict(torch.load("rnn_net.pth"))
att_net.load_state_dict(torch.load("att_net.pth"))
fusion_net.load_state_dict(torch.load("fusion_net.pth"))

# Create a Flask app object
app = Flask(__name__)

# Define the route for the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get the file from the request
    file = request.files["file"]
    # Save the file to a temporary location
    file.save("temp.jpg")
    # Load the image from the file
    image = Image.open("temp.jpg")
    # Transform the image to a tensor
    image = transform_image(image)
    # Move the image to the device
    image = image.to(device)
    # Pass the image to the fusion model and get the logits
    logits = fusion_net(image)
    # Get the predicted class index and name
    class_id = logits.argmax(dim=1).item()
    class_name = index_to_name[class_id]
    # Return the prediction as a JSON response
    return jsonify({"class_id": class_id, "class_name": class_name})
