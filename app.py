import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import laspy
import os
import plotly.graph_objects as go
import tempfile
from scipy.spatial import KDTree
import base64

# Try to import leafmap for visualization options
try:
    import leafmap
    import open3d as o3d
    LEAFMAP_AVAILABLE = True
except ImportError:
    LEAFMAP_AVAILABLE = False

# Import model definition
try:
    from model_definition import DGCNN, knn, get_graph_feature
except ImportError:
    st.error("Error: Could not import model_definition.py")
    st.stop()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
K_NEIGHBORS_FIXED = 20

@st.cache_resource
def load_model(model_path, k_neighbors=K_NEIGHBORS_FIXED):
    if not os.path.exists(model_path):
        st.error(f"Error: Model checkpoint file not found at '{model_path}'.")
        return None

    try:
        model = DGCNN(input_channels=6, output_channels=2, k=k_neighbors).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return None

def run_inference(model, las_data, device, num_points_per_region=1024, sample_fraction=0.1):
    """Perform inference on LAS data using KDTree optimization"""
    x, y, z = las_data.x, las_data.y, las_data.z
    num_all_points = len(x)

    if num_all_points == 0:
        st.warning("No points found in the LAS file.")
        return None, None, None, None, None

    # Extract XYZ coordinates for KDTree
    points_xyz = np.column_stack((x, y, z))
    
    # Handle RGB data
    default_rgb_val = 128
    has_rgb = False
    
    if hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue') and \
       len(las_data.red) == num_all_points:
        
        r_orig, g_orig, b_orig = las_data.red, las_data.green, las_data.blue
        max_val = max(np.max(r_orig), np.max(g_orig), np.max(b_orig), 1)
        
        if max_val > 255:
            # Scale 16-bit to 8-bit for display
            r_display = (r_orig / 65535.0 * 255.0).astype(np.uint8)
            g_display = (g_orig / 65535.0 * 255.0).astype(np.uint8)
            b_display = (b_orig / 65535.0 * 255.0).astype(np.uint8)
            # Normalize for model features (0-1)
            r_feat = (r_orig / 65535.0).astype(np.float32)
            g_feat = (g_orig / 65535.0).astype(np.float32)
            b_feat = (b_orig / 65535.0).astype(np.float32)
        else:
            r_display = r_orig.astype(np.uint8)
            g_display = g_orig.astype(np.uint8)
            b_display = b_orig.astype(np.uint8)
            r_feat = (r_orig / 255.0).astype(np.float32)
            g_feat = (g_orig / 255.0).astype(np.float32)
            b_feat = (b_orig / 255.0).astype(np.float32)
        has_rgb = True
    else:
        # Use default gray
        r_display = g_display = b_display = np.full(num_all_points, default_rgb_val, dtype=np.uint8)
        r_feat = g_feat = b_feat = np.full(num_all_points, 0.5, dtype=np.float32)

    # Features for the model
    all_points_features = np.column_stack((x, y, z, r_feat, g_feat, b_feat))
    predictions = np.zeros(num_all_points, dtype=int)

    # Build KDTree
    st.info("Building KDTree...")
    try:
        kdtree = KDTree(points_xyz)
    except Exception as e:
        st.error(f"Error building KDTree: {e}")
        return None, None, None, None, None

    # Sample points for inference
    sample_size = max(min(int(num_all_points * sample_fraction), num_all_points), 1)
    sampled_indices = np.random.choice(num_all_points, sample_size, replace=False)
    
    st.info(f"Running inference on {sample_size} sampled regions...")
    progress_bar = st.progress(0)
    
    # Run inference
    with torch.no_grad():
        for i, idx in enumerate(sampled_indices):
            center_point_xyz = points_xyz[idx]
            
            # Query KDTree for nearest neighbors
            k_query = min(num_points_per_region + 1, num_all_points)
            try:
                distances, nearest_indices = kdtree.query(center_point_xyz, k=k_query)
            except Exception as e:
                continue
                
            if isinstance(nearest_indices, (int, np.integer)):
                nearest_indices = [nearest_indices]
                
            valid_neighbor_indices = [ni for ni in nearest_indices if ni != idx][:num_points_per_region]
            
            if not valid_neighbor_indices:
                continue
                
            # Get features for the model
            region_features_model = all_points_features[valid_neighbor_indices]
            
            # Center coordinates
            centered_region_features_model = region_features_model.copy()
            centered_region_features_model[:, :3] = region_features_model[:, :3] - center_point_xyz
            
            # Ensure region has exactly num_points_per_region
            current_region_size = centered_region_features_model.shape[0]
            if current_region_size < num_points_per_region:
                num_to_pad = num_points_per_region - current_region_size
                padding = np.repeat(centered_region_features_model[-1:], num_to_pad, axis=0)
                centered_region_features_model = np.vstack((centered_region_features_model, padding))
                
            # Prepare input for model
            inputs = torch.FloatTensor(centered_region_features_model).unsqueeze(0).permute(0, 2, 1).to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, predicted_label = outputs.max(1)
            
            # Mark points as pothole if predicted
            if predicted_label.item() == 1:
                predictions[valid_neighbor_indices] = 1
                
            # Update progress
            progress_bar.progress((i + 1) / sample_size)
            
    progress_bar.empty()
    st.success("Inference complete.")
    num_potholes = np.sum(predictions)
    st.info(f"Found {num_potholes} potential pothole points out of {num_all_points} total points.")
    
    return points_xyz, predictions, r_display, g_display, b_display

def visualize_with_leafmap(points_xyz, r, g, b, predictions=None, backend="pyvista", point_size=1.0):
    """Create visualization with leafmap"""
    if not LEAFMAP_AVAILABLE:
        st.error("Leafmap visualization requires leafmap and open3d packages.")
        return None
        
    # Create a temporary LAS file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Create a new LAS file
        header = laspy.LasHeader(point_format=7)
        las = laspy.LasData(header)
        
        # Set coordinates
        las.x = points_xyz[:, 0]
        las.y = points_xyz[:, 1]
        las.z = points_xyz[:, 2]
        
        # Set colors
        max_color = max(np.max(r), np.max(g), np.max(b))
        if max_color <= 255:
            # Scale from 0-255 to 0-65535 for LAS format
            las.red = (r * 257).astype(np.uint16)
            las.green = (g * 257).astype(np.uint16)
            las.blue = (b * 257).astype(np.uint16)
        else:
            las.red = r.astype(np.uint16)
            las.green = g.astype(np.uint16)
            las.blue = b.astype(np.uint16)
        
        # Set classification if predictions are provided
        if predictions is not None:
            classification = np.ones(len(predictions), dtype=np.uint8)
            classification[predictions == 1] = 7  # Mark potholes as class 7
            las.classification = classification
        
        # Write to file
        las.write(tmp_path)
        
        # Generate visualization with specified backend
        color_options = {"color_column": "classification"} if predictions is not None else {"cmap": "terrain"}
        
        # Generate HTML for visualization
        html = leafmap.view_lidar(
            tmp_path,
            background="white", 
            backend=backend,
            return_as="html",
            point_size=point_size,
            width=800,
            height=600,
            **color_options
        )
        return html
            
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass

def export_las_with_predictions(las_data, predictions, output_path="pothole_results.las"):
    """Export LAS file with pothole classifications"""
    try:
        new_las = laspy.LasData(las_data.header)
        
        # Copy all point data
        for dimension in las_data.point_format.dimension_names:
            setattr(new_las, dimension, getattr(las_data, dimension))
        
        # Add classification field
        classification = np.ones(len(predictions), dtype=np.uint8)
        classification[predictions == 1] = 7  # Mark potholes as class 7
        
        new_las.classification = classification
        new_las.write(output_path)
        return output_path
    except Exception as e:
        st.error(f"Error exporting LAS file: {e}")
        return None

def get_download_link(file_path, link_text):
    """Create download link for file"""
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    filename = os.path.basename(file_path)
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Streamlit App UI
st.set_page_config(layout="wide", page_title="3D Pothole Detection", initial_sidebar_state="expanded")

# Header
st.title("🛣️ 3D Pothole Detection & Visualization")
st.markdown("Upload a `.las` or `.laz` file to detect and visualize potholes.")

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.info(f"Using device: **{str(device).upper()}**")

# Parameters
num_region_points = st.sidebar.slider("Points per Region", 512, 2048, 1024, 128)
inference_sample_fraction = st.sidebar.slider("Inference Sample Fraction", 0.01, 0.5, 0.1, 0.01)
vis_method = st.sidebar.selectbox("Visualization Method", 
                                 ["Leafmap", "Plotly 3D"],
                                 help="Choose visualization method")

# Leafmap specific options
if vis_method == "Leafmap" and LEAFMAP_AVAILABLE:
    leafmap_backend = st.sidebar.selectbox(
        "Leafmap Backend",
        ["pyvista", "open3d", "ipygany", "panel"],
        index=0,
        help="Select the backend for leafmap visualization"
    )
    
    point_size = st.sidebar.slider(
        "Point Size", 
        min_value=0.5, 
        max_value=5.0, 
        value=1.5, 
        step=0.1,
        help="Size of the points in the visualization"
    )
    
    colorby = st.sidebar.radio(
        "Color By",
        ["Original", "Classification (Pothole/Non-Pothole)"],
        index=1,
        help="Choose how to color the points"
    )

# Load model
model = load_model(MODEL_PATH)

# Main UI
tab_upload, tab_results = st.tabs(["📤 Upload & Process", "📊 Results & Visualization"])

with tab_upload:
    st.subheader("📂 Upload LAS/LAZ File")
    uploaded_file = st.file_uploader("Select file:", type=["las", "laz"], label_visibility="collapsed")
    
    if uploaded_file is not None and model is not None:
        st.write("---")
        st.info(f"Processing '{uploaded_file.name}'...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".las") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            las_data = laspy.read(tmp_file_path)
            st.success(f"Successfully read file. Contains **{len(las_data.points):,}** points.")

            # Run inference
            with st.spinner("Running pothole detection... This may take a while."):
                 points_xyz, predictions, r_display, g_display, b_display = run_inference(
                    model, las_data, device,
                    num_points_per_region=num_region_points,
                    sample_fraction=inference_sample_fraction
                 )
                 
                 if points_xyz is not None:
                    # Store results for visualization
                    st.session_state['points_xyz'] = points_xyz
                    st.session_state['predictions'] = predictions
                    st.session_state['r_display'] = r_display
                    st.session_state['g_display'] = g_display
                    st.session_state['b_display'] = b_display
                    st.session_state['las_data'] = las_data
                    st.session_state['filename'] = uploaded_file.name
                    
                    # Success message
                    num_potholes = np.sum(predictions == 1)
                    pothole_percentage = (num_potholes / len(predictions)) * 100
                    
                    st.success("✅ Processing complete! Switch to 'Results & Visualization' tab.")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Points", f"{len(predictions):,}")
                    with col2:
                        st.metric("Pothole Points", f"{num_potholes:,}")
                    with col3:
                        st.metric("Pothole Percentage", f"{pothole_percentage:.2f}%")
                 else:
                    st.error("⚠️ Inference failed. No valid data for visualization.")

        except Exception as e:
            st.error(f"An error occurred while processing the file:")
            st.exception(e)
        finally:
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    elif uploaded_file is not None and model is None:
        st.error("Model could not be loaded. Check console errors.")
    else:
        st.info("👆 Upload a `.las` or `.laz` file to begin.")
        
        # Example images
        st.subheader("Example Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://i.imgur.com/XQGWbJk.gif", caption="Panel Backend")
        with col2:
            st.image("https://i.imgur.com/rL85fbl.gif", caption="Open3D Backend")

with tab_results:
    st.subheader("📊 Visualization Results")
    
    if 'points_xyz' in st.session_state:
        # Export option
        export_col1, export_col2 = st.columns([3, 1])
        with export_col1:
            st.subheader("🔍 Export Results")
        
        with export_col2:
            export_button = st.button("Export as LAS", type="primary")
        
        if export_button:
            with st.spinner("Exporting results to LAS file..."):
                output_path = f"pothole_results_{os.path.basename(st.session_state['filename'])}"
                result_path = export_las_with_predictions(
                    st.session_state['las_data'], 
                    st.session_state['predictions'], 
                    output_path
                )
                if result_path:
                    st.success(f"Results exported successfully")
                    st.markdown(get_download_link(result_path, "📥 Download Results"), unsafe_allow_html=True)
        
        # Visualization
        st.subheader("📊 Point Cloud Visualization")
        
        if vis_method == "Leafmap" and LEAFMAP_AVAILABLE:
            st.markdown(f"Using Leafmap with **{leafmap_backend}** backend")
            
            use_predictions = colorby == "Classification (Pothole/Non-Pothole)"
            
            with st.spinner(f"Generating visualization..."):
                html_content = visualize_with_leafmap(
                    st.session_state['points_xyz'], 
                    st.session_state['r_display'], 
                    st.session_state['g_display'], 
                    st.session_state['b_display'], 
                    predictions=st.session_state['predictions'] if use_predictions else None,
                    backend=leafmap_backend,
                    point_size=point_size
                )
                
                if html_content:
                    st.components.v1.html(html_content, height=700)
                else:
                    st.warning("Could not generate visualization.")
        
        # Display pothole stats
        st.subheader("📈 Pothole Detection Stats")
        num_potholes = np.sum(st.session_state['predictions'] == 1)
        total_points = len(st.session_state['predictions'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Points", f"{total_points:,}")
        with col2:
            st.metric("Pothole Points", f"{num_potholes:,}")
        with col3:
            st.metric("Pothole Percentage", f"{(num_potholes/total_points*100):.2f}%")
            
    else:
        st.info("No data available for visualization. Please upload and process a file first.")

# Help section in sidebar
with st.sidebar.expander("❓ Need Help?"):
    st.markdown("""
    This app uses a Deep Graph CNN to detect potholes in LiDAR point clouds.
    
    **Quick Guide:**
    1. Upload a LAS/LAZ file
    2. Adjust parameters if needed
    3. View results and visualizations
    4. Export classified results
    
    For better visualization, install leafmap:
    ```
    pip install "leafmap[lidar]" open3d
    ```
    """)

st.sidebar.markdown("---")
st.sidebar.info("🚗 Drive Safer with Better Road Analysis 🚗")