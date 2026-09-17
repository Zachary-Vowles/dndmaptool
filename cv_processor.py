import cv2
import numpy as np
import easyocr
from thefuzz import process
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        logger.info("Initializing EasyOCR Model...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

def auto_crop_map(image: np.ndarray):
    """
    Analyzes the uploaded screenshot to find and crop the main map area.
    The map is typically a bright parchment rectangle on a dark background.
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
    
    if (w * h) > (0.10 * img_area):
        cropped = image[y:y+h, x:x+w]
        return cropped, (x, y, w, h)
    else:
        img_h, img_w = image.shape[:2]
        return image, (0, 0, img_w, img_h)

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def extract_modules_and_coordinates(cropped_img: np.ndarray, whitelist: list[str]) -> dict[str, tuple[int, int]]:
    reader = get_ocr_reader()
    processed_img = preprocess_for_ocr(cropped_img)
    raw_results = reader.readtext(processed_img)
    mapped_modules = {}
    
    for bbox, raw_text, confidence in raw_results:
        if confidence < 0.20 or len(raw_text.strip()) < 3:
            continue
        best_match, score = process.extractOne(raw_text, whitelist)
        if score >= 75:
            top_left = bbox[0]
            bottom_right = bbox[2]
            center_x = int((top_left[0] + bottom_right[0]) / 2)
            center_y = int((top_left[1] + bottom_right[1]) / 2)
            mapped_modules[best_match] = (center_x, center_y)
            
    return mapped_modules

def find_player_location(cropped_img: np.ndarray) -> tuple[int, int] | None:
    """Locates the player icon using green color thresholding (#56FF00)."""
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

def analyze_topology(cropped_img: np.ndarray, mapped_modules: dict[str, tuple[int, int]], adjacency_pct: float = 0.18, dark_thresh: int = 65) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
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
                dx = abs(coord_a[0] - coord_b[0])
                dy = abs(coord_a[1] - coord_b[1])
                
                if dx > 1.5 * dy or dy > 1.5 * dx:
                    all_edges.append((mod_a, mod_b))
                    
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

def draw_debug_graph(cropped_img: np.ndarray, mapped_modules: dict[str, tuple[int, int]], all_edges: list, valid_edges: list) -> np.ndarray:
    debug_img = cropped_img.copy()
    for edge in all_edges:
        if edge not in valid_edges and (edge[1], edge[0]) not in valid_edges:
            cv2.line(debug_img, mapped_modules[edge[0]], mapped_modules[edge[1]], (0, 0, 255), 2)
            
    for edge in valid_edges:
        cv2.line(debug_img, mapped_modules[edge[0]], mapped_modules[edge[1]], (0, 255, 0), 3)
        
    for mod_name, coords in mapped_modules.items():
        cv2.circle(debug_img, coords, 6, (255, 0, 0), -1)
        
    return debug_img

def draw_route_on_image(image: np.ndarray, crop_box: tuple, mapped_modules: dict[str, tuple[int, int]], route: list[str]) -> np.ndarray:
    if not route or len(route) < 2:
        return image
        
    output_image = image.copy()
    crop_offset_x, crop_offset_y = crop_box[0], crop_box[1]
    
    global_coords = {}
    for mod_name, (cx, cy) in mapped_modules.items():
        global_coords[mod_name] = (cx + crop_offset_x, cy + crop_offset_y)
        
    for i in range(len(route) - 1):
        node_a = route[i]
        node_b = route[i+1]
        if node_a in global_coords and node_b in global_coords:
            pt1 = global_coords[node_a]
            pt2 = global_coords[node_b]
            cv2.line(output_image, pt1, pt2, (0, 255, 255), 4)
            cv2.circle(output_image, pt1, 8, (0, 0, 255), -1)
            
    if route[-1] in global_coords:
        cv2.circle(output_image, global_coords[route[-1]], 8, (0, 0, 255), -1)
        
    return output_image