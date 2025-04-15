# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import laspy
import os
import plotly.graph_objects as go
from tqdm import tqdm # For progress indication during inference
import tempfile # To handle uploaded file
from scipy.spatial import KDTree # <-- Import KDTree

# --- Import the model class and helpers (Requires model_definition.py in the same directory) ---
try:
    from model_definition import DGCNN, knn, get_graph_feature
except ImportError:
    st.error("Error: Could not import from 'model_definition.py'. Make sure the file exists in the same directory as app.py and contains the DGCNN class and helper functions.")
    st.stop() # Stop execution if model definition is missing


# --- Configuration ---
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FIXED MODEL PATH ---
MODEL_PATH = "best_model.pth"

# --- FIXED MODEL PARAMETER ---
K_NEIGHBORS_FIXED = 20

# --- Model Loading Function ---
@st.cache_resource
def load_model(model_path, k_neighbors=K_NEIGHBORS_FIXED):
    # ... (load_model function remains the same) ...
    if not os.path.exists(model_path):
        st.error(f"Error: Model checkpoint file not found at '{model_path}'.")
        st.error("Please ensure the model file exists at the specified location.")
        return None

    try:
        model = DGCNN(input_channels=6, output_channels=2, k=k_neighbors).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        st.success(f"Model loaded successfully from '{model_path}' (k={k_neighbors})")
        return model
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        st.exception(e)
        return None


# --- Inference Function (OPTIMIZED with KDTree) ---
def run_inference_for_visualization(model, las_data, device, num_points_per_region=1024, sample_fraction=0.1):
    """
    Perform inference on LAS data (Optimized with KDTree) and return points with predictions.
    """
    x, y, z = las_data.x, las_data.y, las_data.z
    num_all_points = len(x)

    if num_all_points == 0:
        st.warning("No points found in the LAS file.")
        return None, None

    # Extract XYZ coordinates for KDTree and features
    points_xyz = np.column_stack((x, y, z))

    # --- Feature Extraction ---
    default_rgb_val = 0.5
    if hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue') and len(las_data.red) == num_all_points:
        r, g, b = las_data.red, las_data.green, las_data.blue
        max_r, max_g, max_b = (np.max(c) if len(c)>0 else 1 for c in (r,g,b))
        max_val = max(max_r, max_g, max_b, 1)
        norm_factor = 65535.0 if max_val > 255 else 255.0
        r = (r / norm_factor).astype(np.float32)
        g = (g / norm_factor).astype(np.float32)
        b = (b / norm_factor).astype(np.float32)
    else:
        st.warning("RGB data not found or incomplete in LAS file. Using default gray color.")
        r = np.full(num_all_points, default_rgb_val, dtype=np.float32)
        g = np.full(num_all_points, default_rgb_val, dtype=np.float32)
        b = np.full(num_all_points, default_rgb_val, dtype=np.float32)

    all_points_features = np.column_stack((x, y, z, r, g, b))
    predictions = np.zeros(num_all_points, dtype=int)

    # --- Build KDTree (build once before loop) ---
    st.info("Building KDTree for efficient neighbor search...")
    try:
        kdtree = KDTree(points_xyz)
    except Exception as e:
        st.error(f"Error building KDTree: {e}. Check if points_xyz is valid.")
        return None, None # Stop if KDTree fails
    st.info("KDTree built.")


    # --- Sampling for Inference ---
    sample_size = int(num_all_points * sample_fraction)
    sample_size = max(min(sample_size, num_all_points), 0)

    if sample_size == 0:
         st.warning("Not enough points to sample for inference based on fraction.")
         return points_xyz, predictions

    sampled_indices = np.random.choice(num_all_points, sample_size, replace=False)
    st.info(f"Running inference on {sample_size} sampled regions...")

    # --- Run Inference on Sampled Regions (using KDTree query) ---
    progress_bar = st.progress(0)
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(sampled_indices):
            center_point_xyz = points_xyz[idx]

            # --- Query KDTree for nearest neighbors ---
            # Query for k neighbors (k=num_points_per_region). Returns distances, indices.
            # Add 1 to k because query point itself is included if it exists in the tree data
            k_query = min(num_points_per_region + 1, num_all_points)
            try:
                distances, nearest_indices = kdtree.query(center_point_xyz, k=k_query)
            except Exception as e:
                 st.warning(f"KDTree query failed for point {idx}: {e}. Skipping region.")
                 continue # Skip this region if query fails

            # If k_query returned a single point (distances is float), make it array
            if isinstance(nearest_indices, (int, np.integer)):
                nearest_indices = [nearest_indices]

            # Remove the query point itself if it's in the results (often the first one)
            # Keep only the required number of neighbors
            valid_neighbor_indices = [ni for ni in nearest_indices if ni != idx][:num_points_per_region]

            if not valid_neighbor_indices:
                 continue # Skip if no valid neighbors found

            # Get features for the region using the found indices
            region_features = all_points_features[valid_neighbor_indices]

            # Center the region coordinates (XYZ) relative to the center point
            centered_region_features = region_features.copy()
            centered_region_features[:, :3] = region_features[:, :3] - center_point_xyz #:3] unnecessary here

            # Ensure region has exactly num_points_per_region by padding if necessary
            current_region_size = centered_region_features.shape[0]
            if current_region_size < num_points_per_region:
                num_to_pad = num_points_per_region - current_region_size
                padding = np.repeat(centered_region_features[-1:], num_to_pad, axis=0)
                centered_region_features = np.vstack((centered_region_features, padding))

            # Prepare input for model: [1, F, N]
            inputs = torch.FloatTensor(centered_region_features).unsqueeze(0).permute(0, 2, 1).to(device)

            # Forward pass
            outputs = model(inputs)
            _, predicted_label = outputs.max(1)

            # If predicted as pothole (label 1), mark points in that original region as pothole
            if predicted_label.item() == 1:
                predictions[valid_neighbor_indices] = 1 # Assign pothole label to neighbors

            # Update progress bar
            progress_bar.progress((i + 1) / sample_size)

    progress_bar.empty()
    st.success("Inference complete.")
    num_potholes = np.sum(predictions)
    st.info(f"Found {num_potholes} potential pothole points out of {num_all_points} total points.")
    return points_xyz, predictions # Return original XYZ and predictions


# --- Visualization Function ---
def plot_point_cloud(points_xyz, predictions):
    # ... (Visualization function remains the same as before) ...
    if points_xyz is None or predictions is None or len(points_xyz) == 0:
        st.warning("Cannot generate plot: No point data available.")
        return None
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    point_colors = np.where(predictions == 1, 'red', 'blue')
    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=1.5, color=point_colors, opacity=0.8),
        name='Points'
    )
    layout = go.Layout(
        title='Pothole Detection Visualization (Red = Pothole)',
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data', bgcolor='rgba(0,0,0,0)'),
        legend=dict(itemsizing='constant'),
        margin=dict(l=10, r=10, b=10, t=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig


# --- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Pothole Detection from LAS")
st.title("🛣️ 3D Pothole Detection from LAS Point Cloud")
st.write("Upload a `.las` file to visualize the point cloud and detected potholes.")
st.sidebar.header("⚙️ Configuration")
st.sidebar.info(f"Using device: {device}")
st.sidebar.info(f"Using fixed k-neighbors: {K_NEIGHBORS_FIXED}")
st.sidebar.info(f"Attempting to load model from: '{MODEL_PATH}'")

# Inference parameters
num_region_points = st.sidebar.slider("Points per Region (Inference)", 512, 4096, 1024, help="Number of points sampled around a center point for inference.")
inference_sample_fraction = st.sidebar.slider("Inference Sample Fraction", 0.01, 1.0, 0.1, 0.01, help="Fraction of total points to use as centers for inference (higher values are more accurate but slower).")

# --- Load Model ---
model = load_model(MODEL_PATH)

# --- Main Area ---
uploaded_file = st.file_uploader("📂 Upload your .las file", type=["las"])

if uploaded_file is not None and model is not None:
    st.write("---")
    st.subheader("Processing Uploaded File...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        las_data = laspy.read(tmp_file_path)
        st.success(f"Successfully read '{uploaded_file.name}'. Contains {len(las_data.points)} points.")
        points_xyz, predictions = run_inference_for_visualization(
            model, las_data, device,
            num_points_per_region=num_region_points,
            sample_fraction=inference_sample_fraction
        )
        if points_xyz is not None:
            st.subheader("📊 Visualization")
            with st.spinner("Generating 3D plot..."):
                 fig = plot_point_cloud(points_xyz, predictions)
                 if fig:
                     st.plotly_chart(fig, use_container_width=True)
                 else:
                     st.warning("Could not generate plot.")
        else:
             st.warning("Inference did not return valid points for visualization.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.exception(e)
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
elif uploaded_file is not None and model is None:
    st.error("Model could not be loaded. Please check the path and file integrity.")
else:
    st.info("Upload a .las file to begin.")
st.write("---")
st.sidebar.info("App uses a DGCNN model for pothole detection.")