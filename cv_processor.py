import cv2
import numpy as np
import easyocr
from thefuzz import process
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the EasyOCR reader instance so it only loads into memory once
_ocr_reader = None

def get_ocr_reader():
    """Lazily initializes and returns the EasyOCR reader to save startup time."""
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initializing EasyOCR Model (This may take a moment on first run)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

def auto_crop_map(image: np.ndarray):
    """
    Analyzes the uploaded screenshot to find and crop the main map area.
    """
    logger.info("Auto-cropping the map from the source image...")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning("No contours found. Returning original image.")
        h, w = image.shape[:2]
        return image, (0, 0, w, h)
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    img_area = image.shape[0] * image.shape[1]
    if (w * h) > (0.10 * img_area):
        cropped = image[y:y+h, x:x+w]
        return cropped, (x, y, w, h)
    else:
        logger.warning("Largest contour is too small to be the map. Returning original image.")
        img_h, img_w = image.shape[:2]
        return image, (0, 0, img_w, img_h)

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Pre-processes the cropped map to enhance the readability of the module names.
    """
    logger.info("Pre-processing image for OCR...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def extract_modules_and_coordinates(cropped_img: np.ndarray, whitelist: list[str]) -> dict[str, tuple[int, int]]:
    """
    Reads the text from the map, fuzzy-matches it against the valid theme modules,
    and calculates the central pixel coordinates for each identified module.
    """
    logger.info("Starting OCR extraction...")
    
    reader = get_ocr_reader()
    processed_img = preprocess_for_ocr(cropped_img)
    
    # We use paragraph=True to automatically group vertically stacked text.
    # This prevents "Demon" and "Gate" from being read as two separate modules.
    raw_results = reader.readtext(processed_img, paragraph=True, x_ths=0.5, y_ths=1.0)
    
    mapped_modules = {}
    
    for result in raw_results:
        # When paragraph=True, EasyOCR returns (bbox, text) instead of (bbox, text, confidence).
        if len(result) == 2:
            bbox, raw_text = result
            confidence = 1.0  # Assign dummy high confidence for grouped text
        else:
            bbox, raw_text, confidence = result
            
        if confidence < 0.20 or len(raw_text.strip()) < 3:
            continue
            
        best_match, score = process.extractOne(raw_text, whitelist)
        
        if score >= 75:
            top_left = bbox[0]
            bottom_right = bbox[2]
            
            center_x = int((top_left[0] + bottom_right[0]) / 2)
            center_y = int((top_left[1] + bottom_right[1]) / 2)
            
            # Handle Duplicate Modules (e.g. "Fallen Forest (2)")
            base_name = best_match
            counter = 2
            final_name = base_name
            while final_name in mapped_modules:
                final_name = f"{base_name} ({counter})"
                counter += 1
                
            mapped_modules[final_name] = (center_x, center_y)
            logger.info(f"Matched: '{raw_text}' -> '{final_name}' (Score: {score}) at ({center_x}, {center_y})")
            
    return mapped_modules

def find_player_location(cropped_img: np.ndarray) -> tuple[int, int] | None:
    """
    Locates the player icon on the cropped map by finding the specific neon green color marker.
    """
    logger.info("Searching for player location using green color thresholding...")
    
    # Convert to HSV color space for much easier color filtering
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    
    # Define range for the bright neon green (#56FF00 roughly)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])
    
    # Create a mask that isolates only the green pixels
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Find contours of the green mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest green blob is the player
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Ensure it's not a single random pixel
        if cv2.contourArea(largest_contour) > 10:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                logger.info(f"Player found at ({cX}, {cY}) using color masking.")
                return (cX, cY)
                
    logger.warning("Player not found. No bright green marker detected.")
    return None

def analyze_topology(cropped_img: np.ndarray, mapped_modules: dict[str, tuple[int, int]], adjacency_pct: float = 0.18, dark_thresh: int = 65) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Determines which modules are connected by open passages.
    """
    logger.info("Analyzing map topology for open passages...")
    
    all_edges = []
    valid_edges = []
    module_names = list(mapped_modules.keys())
    
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    
    img_height, img_width = cropped_img.shape[:2]
    adjacency_threshold = int(img_width * adjacency_pct) 
    
    for i in range(len(module_names)):
        for j in range(i + 1, len(module_names)):
            mod_a = module_names[i]
            mod_b = module_names[j]
            
            coord_a = mapped_modules[mod_a]
            coord_b = mapped_modules[mod_b]
            
            dist = np.sqrt((coord_a[0] - coord_b[0])**2 + (coord_a[1] - coord_b[1])**2)
            
            if dist < adjacency_threshold:
                # Enforce strict Cardinal Directions (No Diagonals)
                dx = abs(coord_a[0] - coord_b[0])
                dy = abs(coord_a[1] - coord_b[1])
                
                tolerance = adjacency_threshold * 0.45 
                is_orthogonal = (dx < tolerance and dy > tolerance) or (dy < tolerance and dx > tolerance)
                
                if is_orthogonal:
                    all_edges.append((mod_a, mod_b))
                    
                    mid_x = (coord_a[0] + coord_b[0]) // 2
                    mid_y = (coord_a[1] + coord_b[1]) // 2
                    
                    box_size = max(3, int(adjacency_threshold * 0.10))
                    x1, x2 = max(0, mid_x - box_size), min(img_width, mid_x + box_size)
                    y1, y2 = max(0, mid_y - box_size), min(img_height, mid_y + box_size)
                    
                    patch = gray[y1:y2, x1:x2]
                    
                    if patch.size > 0:
                        dark_pixel_ratio = np.sum(patch < dark_thresh) / patch.size
                        
                        if dark_pixel_ratio < 0.08: 
                            valid_edges.append((mod_a, mod_b))
                            
    logger.info(f"Topology analysis complete. Found {len(all_edges)} total adjacencies, {len(valid_edges)} open paths.")
    return all_edges, valid_edges

def draw_debug_graph(cropped_img: np.ndarray, mapped_modules: dict[str, tuple[int, int]], all_edges: list, valid_edges: list) -> np.ndarray:
    """Draws the detected nodes and edges on the cropped image for debugging."""
    debug_img = cropped_img.copy()
    
    # Draw all edges first (Blocked = Red)
    for edge in all_edges:
        if edge not in valid_edges and (edge[1], edge[0]) not in valid_edges:
            pt1 = mapped_modules[edge[0]]
            pt2 = mapped_modules[edge[1]]
            cv2.line(debug_img, pt1, pt2, (0, 0, 255), 2)
            
    # Draw valid edges over them (Open = Green)
    for edge in valid_edges:
        pt1 = mapped_modules[edge[0]]
        pt2 = mapped_modules[edge[1]]
        cv2.line(debug_img, pt1, pt2, (0, 255, 0), 3)
        
    # Draw nodes
    for mod_name, coords in mapped_modules.items():
        cv2.circle(debug_img, coords, 6, (255, 0, 0), -1)
        
    return debug_img

def draw_route_on_image(image: np.ndarray, crop_box: tuple, mapped_modules: dict[str, tuple[int, int]], route: list[str], target_nodes: list[str] = None) -> np.ndarray:
    """
    Overlays the calculated path onto the original, uncropped image.
    Targets are Blue, intermediate steps are Red.
    """
    if target_nodes is None:
        target_nodes = []
        
    logger.info("Drawing calculated route on image overlay...")
    
    if not route or len(route) < 2:
        logger.warning("Route is empty or too short to draw.")
        return image
        
    output_image = image.copy()
    crop_offset_x, crop_offset_y = crop_box[0], crop_box[1]
    
    line_color = (0, 255, 255) # Yellow path
    line_thickness = 4
    node_color = (0, 0, 255)   # Red dots for intermediate nodes
    node_radius = 8
    
    global_coords = {}
    for mod_name, (cx, cy) in mapped_modules.items():
        global_coords[mod_name] = (cx + crop_offset_x, cy + crop_offset_y)
        
    # Draw the path
    for i in range(len(route) - 1):
        node_a = route[i]
        node_b = route[i+1]
        
        if node_a in global_coords and node_b in global_coords:
            pt1 = global_coords[node_a]
            pt2 = global_coords[node_b]
            
            cv2.line(output_image, pt1, pt2, line_color, line_thickness)
            
            color = (255, 0, 0) if node_a in target_nodes else node_color
            cv2.circle(output_image, pt1, node_radius, color, -1)
            
    if route[-1] in global_coords:
        color = (255, 0, 0) if route[-1] in target_nodes else node_color
        cv2.circle(output_image, global_coords[route[-1]], node_radius, color, -1)
        
    return output_image