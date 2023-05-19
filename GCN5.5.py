import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
from skimage import feature
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from concurrent.futures import ThreadPoolExecutor, as_completed
import cupy as cp
import os
import numpy as np
import SimpleITK as sitk
import scipy.ndimage
from torch_geometric.nn import GCNConv
# from stellargraph import StellarGraph
from sklearn.decomposition import PCA
from keras.models import load_model
PCA_COMPONENTS = 100

input_features = 100  # replace with the correct number of input features
output_features = 1   # replace with the correct number of output features

class GCN(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_features, 256)  # Update input size here
        self.conv2 = GCNConv(256, 128)
        self.fc = torch.nn.Linear(128, output_features)

    def forward(self, x, edge_index, return_last_layer=False):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        if return_last_layer:
            return x
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        return x


loaded_model = GCN(input_features, output_features)
loaded_model.load_state_dict(torch.load('GCN.pth'))


import pandas as pd

def parse_annotations(annotations_file):
    df = pd.read_csv(annotations_file)
    annotations = {}
    for _, row in df.iterrows():
        seriesuid = row['seriesuid']
        if seriesuid not in annotations:
            annotations[seriesuid] = []
        annotations[seriesuid].append((row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm']))
    return annotations

import numpy as np
import scipy.ndimage as ndimage

def segment_lungs(image):
    # Apply a threshold to the image
    binary_image = image > -320

    # Fill the holes in the binary image
    filled_image = ndimage.binary_fill_holes(binary_image)

    # Remove small connected components
    label_image, num_labels = ndimage.label(filled_image)
    sizes = ndimage.sum(filled_image, label_image, range(num_labels + 1))
    mask_size = sizes < 1000
    remove_pixel = mask_size[label_image]
    filled_image[remove_pixel] = 0

    # Apply morphological operations to clean up the segmentation
    struct = ndimage.generate_binary_structure(3, 2)
    eroded_image = ndimage.binary_erosion(filled_image, structure=struct, iterations=2)
    dilated_image = ndimage.binary_dilation(eroded_image, structure=struct, iterations=2)

    return dilated_image


def load_itk_image(filename):
    itk_image = sitk.ReadImage(filename)
    np_image = sitk.GetArrayFromImage(itk_image)
    spacing = itk_image.GetSpacing()
    origin = itk_image.GetOrigin()
    return np_image, spacing, origin

def normalize_intensity(np_image):
    min_intensity = -1000.0
    max_intensity = 400.0
    np_image[np_image < min_intensity] = min_intensity
    np_image[np_image > max_intensity] = max_intensity
    np_image = (np_image - min_intensity) / (max_intensity - min_intensity)
    return np_image

def resample_image(np_image, spacing, new_spacing=[1, 1, 1]):
    resize_factor = np.array(spacing) / np.array(new_spacing)
    new_shape = np.round(np_image.shape * resize_factor)
    real_resize_factor = new_shape / np_image.shape
    new_spacing = spacing / real_resize_factor
    resampled_image = scipy.ndimage.interpolation.zoom(np_image, real_resize_factor, mode='nearest')
    return resampled_image, new_spacing

def threshold_based_segmentation(np_image, threshold=-400):
    binary_image = np.array(np_image > threshold, dtype=np.int8)
    return binary_image

def preprocess_ct_scan(itk_image):
    np_image, origin, spacing = itk_image
    if not isinstance(np_image, np.ndarray):
        raise ValueError("load_itk_image should return a NumPy array as the first element of the returned tuple.")
    
    # Apply intensity normalization
    normalized_image = normalize_intensity(np_image)

    # Segment the lungs
    segmented_lungs = segment_lungs(normalized_image)
    return segmented_lungs

# Load and preprocess CT scans
ct_scan_directory = os.path.join(os.getcwd(), 'Dataset/vali')
ct_scans = []
series_uid= []
for f in os.listdir(ct_scan_directory):
    if f.endswith(".mhd"):
        itk_image = load_itk_image(os.path.join(ct_scan_directory, f))
        seriesuid = os.path.splitext(f)[0]
        series_uid.append(seriesuid)
        ct_scans.append(itk_image)


preprocessed_ct_scans = []
for itk_image in ct_scans:
    preprocessed_image= preprocess_ct_scan(itk_image)
    preprocessed_ct_scans.append(preprocessed_image)

import os
from torch_geometric.utils import to_networkx

def visualize_and_save_graph(graph, ct_scan_id):
    # Create a NetworkX graph from PyTorch Geometric graph
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.num_nodes))

    # Add edges to the NetworkX graph
    for src, dst in zip(*graph.edge_index):
        nx_graph.add_edge(src.item(), dst.item())

    # Visualize the graph
    pos = nx.spring_layout(nx_graph, seed=42)
    plt.figure(figsize=(10, 10))
    nx.draw(nx_graph, pos, node_size=50, node_color="skyblue", edge_color="black", linewidths=1)
    nx.draw_networkx_labels(nx_graph, pos, labels={i: i for i in range(graph.num_nodes)}, font_size=10)
    plt.axis("off")
    plt.title(f"Graph for CT scan {ct_scan_id}")

    # Save the figure
    output = os.getcwd()
    output_dir = os.path.join(output, 'graph')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure with a valid filename
    plt.savefig(os.path.join(output_dir, f"{ct_scan_id}.png"))
    plt.close()

import cupy as cp

def lbp_gpu(img, points, radius):
    rows, cols = img.shape
    lbp = cp.zeros(img.shape, dtype=cp.int32)
    for i, (row, col) in enumerate(points):
        dx = int(radius * cp.cos(2 * cp.pi * i / len(points)))
        dy = int(radius * cp.sin(2 * cp.pi * i / len(points)))
        shifted_row = row - dy
        shifted_col = col - dx
        shifted_row = cp.array(shifted_row)
        shifted_col = cp.array(shifted_col)
        shifted_row = cp.clip(shifted_row, 0, rows - 1).astype(cp.int32)  # Cast to int32
        shifted_col = cp.clip(shifted_col, 0, cols - 1).astype(cp.int32)  # Cast to int32
        shifted_img = img[shifted_row, shifted_col]
        lbp |= (img >= shifted_img) << i
    return lbp.astype(cp.int32).get()



def calculate_adjacency_matrix(features, threshold=None):
    features = cp.array(features)  # Convert features to CuPy array
    n_samples = features.shape[0]
    adjacency_matrix = cp.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            if i != j:
                distance = cp.linalg.norm(features[i].astype(float) - features[j].astype(float))
                if threshold is not None:
                    adjacency_matrix[i, j] = 1 if distance <= threshold else 0
                else:
                    adjacency_matrix[i, j] = distance

    return adjacency_matrix

import numpy as np

def get_circle_points(radius, num_points=8):
    """
    Returns the coordinates of the pixels in a circle of a given radius.
    
    Args:
        radius (int): the radius of the circle.
        num_points (int): the number of points to sample on the circle.
    
    Returns:
        np.ndarray: an array of shape (num_points, 2) containing the coordinates of the pixels on the circle.
    """
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = radius*np.cos(theta)
    y = radius*np.sin(theta)
    points = np.stack((x, y), axis=1)
    return points


def extract_lbp_features(volume, n_components=100):
    if isinstance(volume, tuple):
        np_volume = volume[0]
    else:
        np_volume = volume
    
    # Calculate LBP features
    lbp_features = []
    for i in range(np_volume.shape[0]):
        img = np_volume[i]
        img = cp.asarray(img)  # Convert img to a CuPy array
        points = get_circle_points(radius=1, num_points=8)
        lbp = lbp_gpu(img, points, 1)
        lbp_features.append(lbp.flatten())
    
    lbp_features = np.vstack(lbp_features)

    # Apply PCA to reduce dimensions
    pca = PCA(n_components=n_components)
    lbp_features_reduced = pca.fit_transform(lbp_features)

    return lbp_features_reduced


import networkx as nx

import torch
from torch_geometric.data import Data

def create_graph_from_features(features, adjacency_matrix):
    edge_index = torch.tensor(adjacency_matrix.nonzero(), dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data

from torch_geometric.utils import from_networkx

import networkx as nx

import networkx as nx

def create_networkx_graph_from_adjacency_matrix(adjacency_matrix):
    # Create a NetworkX graph from the adjacency matrix
    G = nx.Graph()
    n, m = adjacency_matrix.shape
    for i in range(n):
        for j in range(m):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)
    return G

from torch_geometric.data import Data

def create_graph_from_features(lbp_features, adjacency_matrix):
    # Create a NetworkX graph
    G = create_networkx_graph_from_adjacency_matrix(adjacency_matrix)
    
    # Add the LBP features as node attributes
    x = torch.tensor(lbp_features, dtype=torch.float)
    
    # Get the edge indices
    edge_index = []
    for edge in G.edges:
        edge_index.append(edge[0])
        edge_index.append(edge[1])
    edge_index = torch.tensor(edge_index, dtype=torch.long).reshape(2, -1)

    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    return data

def process_ct_scan(ct_scan_data):

    lbp_features = extract_lbp_features(ct_scan_data)
    
    # Create graph
    adjacency_matrix = calculate_adjacency_matrix(lbp_features)
    edge_index = torch.tensor(cp.array(adjacency_matrix.nonzero()), dtype=torch.long).t().contiguous()
    x = torch.tensor(lbp_features, dtype=torch.float)
    graph=create_graph_from_features(lbp_features,adjacency_matrix)
    return graph
    
print("CT scans preprocessed successfully.")


ct_scan_graphs = []
for i, ct_scan_data in enumerate(preprocessed_ct_scans):
    graph= process_ct_scan(ct_scan_data)
    ct_scan_graphs.append(graph)

counter = 0
for i, graph in enumerate(ct_scan_graphs):
    # Print edge information
    visualize_and_save_graph(graph, f"ct_scan_{series_uid[counter]}")
    counter += 1
    
def predict_nodules(graph, model):
    # Prepare the input for the model
    edge_index = graph.edge_index  # add this line
    node_features = graph.x
    
    # Predict nodules using the trained model
    with torch.no_grad():
        model.eval()
        predictions = model(node_features, edge_index)  # modify this line
        predictions = torch.squeeze(predictions)

    # You may need to adjust the threshold depending on the model's output
    threshold = 0.3
    nodule_indices = torch.nonzero(predictions > threshold, as_tuple=True)[0]

    return nodule_indices


def get_nodule_coordinates(ct_scan_data, nodule_indices, spacing, origin):
    nodule_coordinates = []
    
    # Get the coordinates in the original image space
    for index in nodule_indices:
        coord_z, coord_y, coord_x = np.unravel_index(index.item(), ct_scan_data.shape)
        coord_x = coord_x * spacing[0] + origin[0]
        coord_y = coord_y * spacing[1] + origin[1]
        coord_z = coord_z * spacing[2] + origin[2]
        
        nodule_coordinates.append((coord_x, coord_y, coord_z))

    return nodule_coordinates

from sklearn.cluster import KMeans

for ct_scan_data, graph, itk_image,i in zip(preprocessed_ct_scans, ct_scan_graphs, ct_scans, series_uid):
    print("CT Scan:"+ i)
    nodule_indices = predict_nodules(graph, loaded_model)
    spacing, origin = itk_image[1], itk_image[2]

    if len(nodule_indices) > 0:
        nodule_coordinates = get_nodule_coordinates(ct_scan_data, nodule_indices, spacing, origin)

        # Cluster nodules within a certain distance
        coords_array = np.array(nodule_coordinates)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(coords_array)
        cluster_labels = kmeans.labels_

        # Get the centroid coordinates for each cluster
        centroids = []
        for label in set(cluster_labels):
            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_coords = coords_array[cluster_indices]
            centroid = np.mean(cluster_coords, axis=0)
            centroids.append(centroid.tolist())

        # Print the coordinates of the nodules
        if len(centroids) > 0:
            print("Nodules detected at the following coordinates:")
            for coord in centroids:
                print(coord)
        else:
            print("No nodules detected.")
    else:
        print("No nodules detected.")

