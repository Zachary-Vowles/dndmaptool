import streamlit as st
import cv2
import numpy as np
import json
from PIL import Image
import cv_processor
import pathfinder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    with open("map_data.json", "r") as f:
        return json.load(f)

def main():
    st.set_page_config(page_title="Map Helper", layout="wide")
    st.title("🗺️ Map Helper Application")

    map_data = load_data()
    themes = list(map_data.keys())

    # --- SIDEBAR UI ---
    st.sidebar.header("Configuration")
    selected_theme = st.sidebar.selectbox("Select Map Theme", themes)
    
    targets = map_data[selected_theme]["targets"]
    target_categories = list(targets.keys())
    selected_category = st.sidebar.selectbox("Select Points of Interest", target_categories)
    
    uploaded_file = st.sidebar.file_uploader("Upload Map Screenshot", type=["png", "jpg", "jpeg"])
    
    st.sidebar.markdown("---")
    st.sidebar.header("Advanced Tuning")
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
    distance_slider = st.sidebar.slider("Adjacency Distance", 0.10, 0.30, 0.18, 0.01)
    dark_thresh_slider = st.sidebar.slider("Wall Darkness Threshold", 10, 150, 65, 5)

    # --- MAIN VIEW ---
    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        image_np = np.array(image_pil)
        
        # Convert RGB (Pillow) to BGR (OpenCV)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if st.sidebar.button("Calculate Route"):
            with st.spinner("Processing map... This may take a few seconds."):
                try:
                    # 1. Crop
                    cropped_img, crop_box = cv_processor.auto_crop_map(image_np)
                    
                    # 2. OCR & Mapping
                    whitelist = map_data[selected_theme]["modules"]
                    mapped_modules = cv_processor.extract_modules_and_coordinates(cropped_img, whitelist)
                    
                    if not mapped_modules:
                        st.error("Could not read any modules. Try a clearer screenshot.")
                        return
                        
                    # 3. Topology (Using Sliders)
                    all_edges, valid_edges = cv_processor.analyze_topology(
                        cropped_img, mapped_modules, 
                        adjacency_pct=distance_slider, 
                        dark_thresh=dark_thresh_slider
                    )
                    
                    # 4. Find Player
                    player_coords = cv_processor.find_player_location(cropped_img)
                    if not player_coords:
                        st.error("Could not find the player icon (bright green marker).")
                        return

                    # 5. Routing
                    G_smart = pathfinder.build_map_graph(mapped_modules, valid_edges)
                    G_direct = pathfinder.build_map_graph(mapped_modules, all_edges)
                    
                    start_node = pathfinder.find_nearest_node_to_player(G_smart, player_coords)
                    
                    # Resolve Target Nodes (including duplicates like "Fallen Forest (2)")
                    base_targets = targets[selected_category]
                    actual_targets = []
                    for node in mapped_modules.keys():
                        base_name = node.split(" (")[0] # Strips the " (2)" off if it exists
                        if base_name in base_targets:
                            actual_targets.append(node)
                    
                    smart_route = pathfinder.calculate_optimal_route(G_smart, start_node, actual_targets)
                    direct_route = pathfinder.calculate_optimal_route(G_direct, start_node, actual_targets)

                    # 6. Drawing
                    smart_img = cv_processor.draw_route_on_image(image_np, crop_box, mapped_modules, smart_route, actual_targets)
                    direct_img = cv_processor.draw_route_on_image(image_np, crop_box, mapped_modules, direct_route, actual_targets)
                    
                    # 7. Render UI Tabs
                    tab1, tab2, tab3 = st.tabs(["Direct Route (Ignores Walls)", "Smart Route (Avoids Walls)", "Debug View"])
                    
                    with tab1:
                        st.markdown("### 🗺️ Route Legend")
                        st.markdown("🔵 **Target Module** &nbsp; | &nbsp; 🔴 **Intermediate Path Node** &nbsp; | &nbsp; 🟡 **Calculated Route**")
                        st.markdown(f"**Path Order:** `{' -> '.join(direct_route)}`")
                        st.image(cv2.cvtColor(direct_img, cv2.COLOR_BGR2RGB), width="stretch")
                    
                    with tab2:
                        st.markdown("### 🗺️ Route Legend")
                        st.markdown("🔵 **Target Module** &nbsp; | &nbsp; 🔴 **Intermediate Path Node** &nbsp; | &nbsp; 🟡 **Calculated Route**")
                        st.markdown(f"**Path Order:** `{' -> '.join(smart_route)}`")
                        st.image(cv2.cvtColor(smart_img, cv2.COLOR_BGR2RGB), width="stretch")
                        
                    with tab3:
                        if debug_mode:
                            st.write("### Detected Modules List:")
                            st.write(", ".join(sorted(mapped_modules.keys())))
                            
                            debug_graph = cv_processor.draw_debug_graph(cropped_img, mapped_modules, all_edges, valid_edges)
                            st.image(cv2.cvtColor(debug_graph, cv2.COLOR_BGR2RGB), caption="Green = Open Path, Red = Blocked/Diagonal", width="stretch")
                        else:
                            st.info("Check 'Enable Debug Mode' in the sidebar to see the internal graph topography.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
        else:
            st.image(image_pil, caption="Uploaded Screenshot", width="stretch")
    else:
        st.info("Please upload a map screenshot using the sidebar to begin.")

if __name__ == "__main__":
    main()