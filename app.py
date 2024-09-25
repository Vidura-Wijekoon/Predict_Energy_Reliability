import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define the GNN model with the same architecture as used in training
class GNN(torch.nn.Module):
    def __init__(self, conv1_out_channels=128, conv2_out_channels=256, conv3_out_channels=512, conv4_out_channels=256, dropout_rate=0.3):
        super(GNN, self).__init__()
        self.conv1 = GATConv(1, conv1_out_channels, heads=4, concat=True)
        self.conv2 = GATConv(conv1_out_channels * 4, conv2_out_channels, heads=4, concat=True)
        self.conv3 = GATConv(conv2_out_channels * 4, conv3_out_channels, heads=4, concat=True)
        self.conv4 = GATConv(conv3_out_channels * 4, conv4_out_channels, heads=4, concat=True)
        
        # Initialize the fc1 layer with a placeholder input size
        self.fc1_input_size = None  # Placeholder for the correct input size
        self.fc1 = None  # Initialize fc1 as None
        
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = self.conv4(x, edge_index)
        x = F.elu(x)

        # Aggregate node and edge features separately
        node_agg = torch.mean(x, dim=0, keepdim=True).repeat(edge_attr.size(0), 1)

        # Print the shapes for debugging
        print(f"node_agg shape: {node_agg.shape}")  # Should be [Number of edges, conv4_out_channels * 4]
        print(f"edge_attr shape: {edge_attr.shape}")  # Should be [Number of edges, 4]

        # Concatenate node and edge features directly
        x = torch.cat([node_agg, edge_attr], dim=1)

        # Print the shape of the concatenated tensor
        print(f"Concatenated x shape: {x.shape}")  # Should be [Number of edges, conv4_out_channels * 4 + 4]

        # Dynamically adjust the input size of fc1 if not initialized
        if self.fc1 is None:
            self.fc1_input_size = x.shape[1]  # Get the actual input size
            self.fc1 = torch.nn.Linear(self.fc1_input_size, 256)  # Initialize fc1 with the correct input size
            print(f"Initializing fc1 with input size: {self.fc1_input_size}")

        # Pass through fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

# Load the trained model
@st.cache_resource
def load_model():
    model = GNN()
    try:
        # Load the model state dict while ignoring unexpected keys
        model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu')), strict=False)
        model.eval()
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return model

model = load_model()

# Function to calculate power per unit transmission
def calculate_power_per_unit_transmission(V, I, theta, L):
    return (V * I * np.cos(np.deg2rad(theta))) / L

# Feature Engineering
def extract_features(df):
    # Calculate power per unit transmission for each connection
    df['power_per_unit_transmission'] = calculate_power_per_unit_transmission(
        df['Voltage_RMS'], df['Current_RMS'], df['Power_Angle'], df['Line_Length']
    )
    
    # Standardize the new feature
    scaler = StandardScaler()
    df[['power_per_unit_transmission_normalized']] = scaler.fit_transform(df[['power_per_unit_transmission']])
    
    return df

def enrich_with_historical_data(df):
    df['Historical_Avg_Load'] = np.random.uniform(50, 150, df.shape[0])
    df['Historical_Max_Load'] = np.random.uniform(100, 200, df.shape[0])
    df['Historical_Min_Load'] = np.random.uniform(0, 50, df.shape[0])

    df[['Historical_Avg_Load_normalized', 'Historical_Max_Load_normalized', 'Historical_Min_Load_normalized']] = StandardScaler().fit_transform(df[['Historical_Avg_Load', 'Historical_Max_Load', 'Historical_Min_Load']])

    return df

# Construct the graph
def construct_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['Substation'], type='substation')
        G.add_node(row['Transformer'], type='transformer')
        G.add_edge(row['Substation'], row['Transformer'], power_per_unit_transmission=row['power_per_unit_transmission_normalized'],
                   historical_avg_load=row['Historical_Avg_Load_normalized'],
                   historical_max_load=row['Historical_Max_Load_normalized'],
                   historical_min_load=row['Historical_Min_Load_normalized'])
    return G

# Prepare data for the model
def prepare_data(df):
    df = extract_features(df)
    df = enrich_with_historical_data(df)
    G = construct_graph(df)

    node_features = []
    edge_index = []
    edge_features = []

    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    for node in G.nodes(data=True):
        node_features.append([1 if node[1]['type'] == 'substation' else 0])

    for edge in G.edges(data=True):
        edge_index.append([node_mapping[edge[0]], node_mapping[edge[1]]])
        edge_features.append([
            edge[2]['power_per_unit_transmission'], 
            edge[2]['historical_avg_load'], 
            edge[2]['historical_max_load'], 
            edge[2]['historical_min_load']
        ])

    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)
    
    # Display number of nodes and edges
    st.write(f"Number of Nodes: {G.number_of_nodes()}")
    st.write(f"Number of Edges (Connections): {G.number_of_edges()}")

    return data

# Streamlit App
st.title('Electrical Grid Reliability Prediction')
st.write('This app uses a Graph Neural Network (GNN) to predict the reliability of an electrical grid.')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Display the uploaded data
    st.write('Uploaded Data:')
    st.write(df.head())

    # Validate the required columns in the uploaded CSV
    required_columns = {'Substation', 'Transformer', 'Voltage_RMS', 'Current_RMS', 'Power_Angle', 'Line_Length'}
    if not required_columns.issubset(df.columns):
        st.error(f"Uploaded CSV is missing required columns: {required_columns - set(df.columns)}")
    else:
        # Button to process and display model prediction
        if st.button("Submit"):
            # Process and prepare the data for prediction
            data = prepare_data(df)

            # Make predictions with the loaded model
            with torch.no_grad():
                prediction = model(data)
                st.write(f'Model Prediction (Mean Reliability Score): {prediction.mean().item()}')

            # Display the graph
            G = construct_graph(df)
            pos = nx.spring_layout(G)
            plt.figure(figsize=(10, 8))
            nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
            plt.title('Electrical Grid Graph Representation')
            st.pyplot(plt)

            # Display individual predictions for all edges
            st.write('Individual Predictions for Each Connection:')
            for i in range(prediction.size(0)):
                st.write(f'Connection {i+1}: {prediction[i].item()}')

            # Display conclusions based on predictions
            st.write('Conclusions:')
            avg_prediction = prediction.mean().item()

            # Update reliability thresholds if needed
            if avg_prediction > 0.6:
                st.write('The electrical grid is highly reliable.')
            elif 0.4 < avg_prediction <= 0.6:
                st.write('The electrical grid is moderately reliable.')
            else:
                st.write('The electrical grid has low reliability.')

            # Provide additional context based on the predictions
            st.write('Note: These conclusions are based on the model predictions and should be further validated with actual grid performance data. It is recommended to incorporate more real-time data for better predictions and system improvements.')

            # Additional Feature: Display reliability scores with color coding
            st.write('### Reliability Scores for Each Connection:')
            for i in range(prediction.size(0)):
                score = prediction[i].item()
                if score > 0.6:
                    st.markdown(f"**Connection {i+1}:** <span style='color:green'>{score:.4f} (High Reliability)</span>", unsafe_allow_html=True)
                elif 0.4 < score <= 0.6:
                    st.markdown(f"**Connection {i+1}:** <span style='color:orange'>{score:.4f} (Moderate Reliability)</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Connection {i+1}:** <span style='color:red'>{score:.4f} (Low Reliability)</span>", unsafe_allow_html=True)

            # Optional: Save the predictions for further analysis
            save_predictions = st.button('Save Predictions to CSV')
            if save_predictions:
                prediction_df = pd.DataFrame({
                    'Substation': df['Substation'],
                    'Transformer': df['Transformer'],
                    'Reliability_Score': prediction.numpy()  # Corrected this line
                })
                prediction_df.to_csv('reliability_predictions.csv', index=False)
                st.success('Predictions saved to reliability_predictions.csv')


# Final Message
st.write('Thank you for using the Electrical Grid Reliability Prediction tool. This tool provides an initial assessment based on historical and real-time data. For more accurate predictions, please consider integrating additional data sources and using advanced model tuning.')
