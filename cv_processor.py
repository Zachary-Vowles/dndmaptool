import cv2
import numpy as np
import easyocr
from thefuzz import process
import logging
from typing import List, Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global EasyOCR Reader cache to avoid reloading the model per frame
_ocr_reader: Optional[easyocr.Reader] = None


def get_ocr_reader() -> easyocr.Reader:
    """Initializes and returns the singleton instance of the EasyOCR Reader."""
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initializing EasyOCR Model...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader


def auto_crop_map(image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Analyzes the uploaded screenshot to locate and crop the main parchment map area.
    Returns the cropped sub-image and its bounding box tuple (x, y, w, h).
    """
    logger.info("Auto-cropping the map from the source image...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = image.shape[:2]
        return image, (0, 0, w, h)
        
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    img_area = image.shape[0] * image.shape[1]
    
    # Verify that the bounding area is substantial enough to be the map board
    if (w * h) > (0.10 * img_area):
        cropped = image[y:y+h, x:x+w]
        return cropped, (x, y, w, h)
    else:
        img_h, img_w = image.shape[:2]
        return image, (0, 0, img_w, img_h)


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Enhances image contrast using CLAHE to prepare text regions for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def extract_modules_and_coordinates(
    cropped_img: np.ndarray, 
    whitelist: List[str]
) -> Dict[str, Tuple[int, int]]:
    """
    Runs OCR on the cropped map, fuzzy matches text against the theme whitelist,
    deduplicates identical coordinates, and appends instance counters for duplicate room names.
    """
    reader = get_ocr_reader()
    processed_img = preprocess_for_ocr(cropped_img)
    raw_results = reader.readtext(processed_img)
    mapped_modules: Dict[str, Tuple[int, int]] = {}
    
    for bbox, raw_text, confidence in raw_results:
        # Filter low confidence or very short OCR noise
        if confidence < 0.20 or len(raw_text.strip()) < 3:
            continue
            
        best_match, score = process.extractOne(raw_text, whitelist)
        
        # Validation score threshold
        if score >= 75:
            top_left = bbox[0]
            bottom_right = bbox[2]
            center_x = int((top_left[0] + bottom_right[0]) / 2)
            center_y = int((top_left[1] + bottom_right[1]) / 2)
            
            # Prevent duplicate reads for the same coordinate box
            is_duplicate_coord = False
            for existing_coords in mapped_modules.values():
                if abs(existing_coords[0] - center_x) < 15 and abs(existing_coords[1] - center_y) < 15:
                    is_duplicate_coord = True
                    break
            
            if is_duplicate_coord:
                continue

            # Handle duplicate names on the map by appending (2), (3), etc.
            original_match = best_match
            counter = 2
            while best_match in mapped_modules:
                best_match = f"{original_match} ({counter})"
                counter += 1
                
            mapped_modules[best_match] = (center_x, center_y)
            logger.info(f"Matched: '{raw_text}' -> '{best_match}' (Score: {score}) at ({center_x}, {center_y})")
            
    return mapped_modules


def find_player_location(cropped_img: np.ndarray) -> Optional[Tuple[int, int]]:
    """Locates the player icon using HSV green color thresholding (#56FF00 neon green)."""
    logger.info("Searching for player location using green color thresholding...")
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    
    # Target bright neon green
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            logger.info(f"Player found at ({center_x}, {center_y}) using color masking.")
            return (center_x, center_y)
            
    logger.warning("Player not found. No bright green marker detected.")
    return None


def analyze_topology(
    cropped_img: np.ndarray, 
    mapped_modules: Dict[str, Tuple[int, int]], 
    adjacency_pct: float = 0.18, 
    dark_thresh: int = 65
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Constructs graph edges between nodes based on orthogonal adjacency.
    Separates edges into 'all_edges' (unobstructed & obstructed) and 'valid_edges' (passable pathways).
    """
    all_edges: List[Tuple[str, str]] = []
    valid_edges: List[Tuple[str, str]] = []
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
                # Enforce strict Cardinal Directions (horizontal or vertical, no diagonals)
                dx = abs(coord_a[0] - coord_b[0])
                dy = abs(coord_a[1] - coord_b[1])
                tolerance = adjacency_threshold * 0.45 
                is_orthogonal = (dx < tolerance and dy > tolerance) or (dy < tolerance and dx > tolerance)
                
                if is_orthogonal:
                    # 1. Add to the direct connectivity graph (ignoring walls)
                    all_edges.append((mod_a, mod_b))
                    
                    # 2. Check the midpoint between nodes for dark pixels indicating walls
                    mid_x = (coord_a[0] + coord_b[0]) // 2
                    mid_y = (coord_a[1] + coord_b[1]) // 2
                    box_size = int(adjacency_threshold * 0.15) 
                    x1, x2 = max(0, mid_x - box_size), min(img_width, mid_x + box_size)
                    y1, y2 = max(0, mid_y - box_size), min(img_height, mid_y + box_size)
                    
                    patch = gray[y1:y2, x1:x2]
                    if patch.size > 0:
                        dark_pixel_ratio = np.sum(patch < dark_thresh) / patch.size
                        if dark_pixel_ratio < 0.08: 
                            valid_edges.append((mod_a, mod_b))
                            
    return all_edges, valid_edges


def draw_debug_graph(
    cropped_img: np.ndarray, 
    mapped_modules: Dict[str, Tuple[int, int]], 
    all_edges: List[Tuple[str, str]], 
    valid_edges: List[Tuple[str, str]]
) -> np.ndarray:
    """Renders visual debug representation of open paths (green) vs blocked walls/diagonals (red)."""
    debug_img = cropped_img.copy()
    valid_edge_set = set(valid_edges) | {(b, a) for a, b in valid_edges}
    
    # Red lines for blocked connections
    for edge in all_edges:
        if edge not in valid_edge_set:
            cv2.line(debug_img, mapped_modules[edge[0]], mapped_modules[edge[1]], (0, 0, 255), 2)
            
    # Green lines for valid navigable paths
    for edge in valid_edges:
        cv2.line(debug_img, mapped_modules[edge[0]], mapped_modules[edge[1]], (0, 255, 0), 3)
        
    # Blue dots for detected module nodes
    for mod_name, coords in mapped_modules.items():
        cv2.circle(debug_img, coords, 6, (255, 0, 0), -1)
        
    return debug_img


def draw_route_on_image(
    image: np.ndarray, 
    crop_box: Tuple[int, int, int, int], 
    mapped_modules: Dict[str, Tuple[int, int]], 
    route: List[str], 
    target_nodes: Optional[List[str]] = None
) -> np.ndarray:
    """
    Overlays the calculated path onto the original full-size image.
    Targets are highlighted in blue, while intermediate waypoints are marked in red.
    """
    if target_nodes is None:
        target_nodes = []
        
    if not route or len(route) < 2:
        return image
        
    output_image = image.copy()
    crop_offset_x, crop_offset_y = crop_box[0], crop_box[1]
    
    global_coords = {}
    for mod_name, (cx, cy) in mapped_modules.items():
        global_coords[mod_name] = (cx + crop_offset_x, cy + crop_offset_y)
        
    line_color = (0, 255, 255)       # Yellow route line
    line_thickness = 4
    node_radius = 8
    intermediate_color = (0, 0, 255) # Red for normal traversal waypoints
    target_color = (255, 0, 0)       # Blue for target POI rooms
    
    for i in range(len(route) - 1):
        node_a = route[i]
        node_b = route[i + 1]
        
        if node_a in global_coords and node_b in global_coords:
            pt1 = global_coords[node_a]
            pt2 = global_coords[node_b]
            
            # Draw route line
            cv2.line(output_image, pt1, pt2, line_color, line_thickness)
            
            # Draw waypoint node
            node_color = target_color if node_a in target_nodes else intermediate_color
            cv2.circle(output_image, pt1, node_radius, node_color, -1)
            
    # Draw destination node
    if route[-1] in global_coords:
        last_node = route[-1]
        last_color = target_color if last_node in target_nodes else intermediate_color
        cv2.circle(output_image, global_coords[last_node], node_radius, last_color, -1)
        
    return output_image