import heapq
import json
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from typing import List, Dict

# ----------------- Load Graph -----------------
with open("hallway_graph.json") as f:
    graph = json.load(f)

adjacency = {}
for edge in graph["edges"]:
    adjacency.setdefault(edge["from"], []).append((
        edge["to"],
        float(edge["weight"]),  # cast to float
        edge.get("instruction", [])
    ))

# ----------------- FastAPI App -----------------
app = FastAPI()

# Serve the 'static' folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- QR Constants -----------------
KNOWN_QR_WIDTH_CM = 10
FOCAL_LENGTH = 667

# ----------------- Navigation Sessions -----------------
nav_sessions = {}

# ----------------- Models -----------------
class NavigationRequest(BaseModel):
    start: str
    end: str
    target_color: str = None  # optional for color scan

# ----------------- Distance Estimation -----------------
def estimate_distance(bbox_width_px, focal_length_px=498.24):
    """
    Estimate distance to QR code in meters using pinhole camera model.
    
    Args:
        bbox_width_px (float): Width of detected QR code in pixels.
        focal_length_px (float): Focal length in pixels (default: calibrated).
    
    Returns:
        float: Estimated distance in meters.
    """
    # --- Constants ---
    KNOWN_QR_WIDTH_CM = 15.0        # QR code width in cm (updated)
    MAX_DISTANCE_M = 10.0           # Max distance in meters
    MIN_DISTANCE_M = 0.1            # Min distance in meters
    SAFETY_FACTOR = 1.05            # Safety factor (multiplier for robustness)
    
    if bbox_width_px <= 0:
        return float('inf')

    # Calculate raw distance in meters
    distance_m = (KNOWN_QR_WIDTH_CM * focal_length_px) / bbox_width_px / 100.0

    # Apply safety factor
    distance_m *= SAFETY_FACTOR

    # Clamp distance to within the allowed min/max range
    distance_m = max(MIN_DISTANCE_M, min(distance_m, MAX_DISTANCE_M))

    return distance_m

# handles HSV-based color segmentation for red, green, and blue
def get_color_mask(hsv, color, debug=False):
    """Return a mask for the desired color in HSV. Saves debug mask if enabled."""
    mask = None

    if color == "red":
        lower1, upper1 = (0, 120, 70), (10, 255, 255)
        lower2, upper2 = (170, 120, 70), (180, 255, 255)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    elif color == "green":
        lower, upper = (36, 50, 70), (89, 255, 255)
        mask = cv2.inRange(hsv, lower, upper)

    elif color == "blue":
        lower, upper = (90, 50, 70), (128, 255, 255)
        mask = cv2.inRange(hsv, lower, upper)

    if debug and mask is not None:
        filename = f"debug_mask_{color}.jpg"
        cv2.imwrite(filename, mask)
        print(f"[DEBUG] Saved color mask: {filename}")

    return mask

def decode_qr_from_frame(frame, target_color=None, debug=False):
    """Detect QR codes in frame, optionally filtered by color. Debug mode saves intermediate outputs."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if target_color:
        mask = get_color_mask(hsv, target_color.lower())
        if mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=mask)
            if debug:
                cv2.imwrite(f"debug_mask_{target_color}.jpg", mask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imwrite("debug_gray.jpg", gray)

    decoded_objs = decode(gray)

    results = []
    for obj in decoded_objs:
        rect = obj.rect
        results.append({
            "data": obj.data.decode("utf-8"),
            "rect": [rect.left, rect.top, rect.width, rect.height],
            "color": target_color if target_color else "any"
        })

        if debug:
            print(f"[DEBUG] QR Data: {obj.data.decode('utf-8')}, Rect: {rect}")

    if debug:
        # Draw rectangles and labels on frame
        for obj in results:
            x, y, w, h = obj['rect']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, obj['data'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite("debug_detected.jpg", frame)

    return results

import heapq

def dijkstra(graph_adj, start, end):
    """Return step-by-step path as (from, to, instructions)."""
    queue = [(0, start, [])]  # cost, node, path_edges
    visited = set()

    while queue:
        cost, node, path_edges = heapq.heappop(queue)
        if node == end:
            return path_edges  # return list of edges
        if node in visited:
            continue
        visited.add(node)

        for neighbor, weight, steps in graph_adj.get(node, []):
            if neighbor not in visited:
                heapq.heappush(queue, (
                    cost + weight,
                    neighbor,
                    path_edges + [(node, neighbor, steps)]
                ))

    return None

# ----------------- Endpoints -----------------
@app.post("/process_frame/")
async def process_frame(file: UploadFile = File(...), target_color: str = Form(None)):
    """
    Detect QR code in a frame, optionally filtered by color.
    Returns robust, weighted navigation instructions.
    """
    debug = True  # Toggle False in production

    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "error", "message": "Failed to decode image"}

    if debug:
        cv2.imwrite("debug_raw.jpg", frame)

    response_data = {
        "status": "waiting",
        "instructions": [],
        "qr_data": None,
        "estimated_distance": None
    }

    decoded_objs = decode_qr_from_frame(frame, target_color, debug=debug)

    # Draw debug rectangles
    for obj in decoded_objs:
        x, y, w, h = obj['rect']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, obj['data'], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if debug:
        cv2.imwrite("debug_detected.jpg", frame)
        print(f"[DEBUG] Total QR codes detected: {len(decoded_objs)}")

    if decoded_objs:
        nearest_obj = max(decoded_objs, key=lambda x: x['rect'][2])
        x, y, w, h = nearest_obj['rect']

        distance_m = estimate_distance(w)
        response_data["estimated_distance"] = distance_m

        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2
        center_x = x + w // 2
        center_y = y + h // 2

        weighted_instructions = []

        if distance_m > 0.5:
            # Horizontal guidance (weight: how far from center)
            offset_x = center_x - frame_center_x
            if abs(offset_x) > frame.shape[1] * 0.3:
                weighted_instructions.append({"action": "move_sharply_right" if offset_x > 0 else "move_sharply_left", "weight": abs(offset_x)})
            elif abs(offset_x) > frame.shape[1] * 0.1:
                weighted_instructions.append({"action": "move_slightly_right" if offset_x > 0 else "move_slightly_left", "weight": abs(offset_x)})

            # Vertical guidance (weight: how far from center)
            offset_y = center_y - frame_center_y
            if abs(offset_y) > frame.shape[0] * 0.2:
                weighted_instructions.append({"action": "tilt_down" if offset_y > 0 else "tilt_up", "weight": abs(offset_y)})

            # Forward movement (weight: distance)
            if distance_m > 1.5:
                weighted_instructions.append({"action": "move_forward_fast", "weight": distance_m})
            elif distance_m > 0.8:
                weighted_instructions.append({"action": "move_forward", "weight": distance_m})
            else:
                weighted_instructions.append({"action": "approach_slowly", "weight": distance_m})

            # Sort instructions by weight (highest priority first)
            weighted_instructions.sort(key=lambda x: x["weight"], reverse=True)

            response_data["status"] = "pre_scan"
        else:
            qr_data = nearest_obj['data'].strip().upper()
            response_data["status"] = "scanned"
            response_data["qr_data"] = qr_data

        response_data["instructions"] = weighted_instructions

        if debug:
            print(f"[DEBUG] Nearest QR: {nearest_obj['data']}")
            print(f"[DEBUG] Estimated distance: {distance_m:.2f} m")
            print(f"[DEBUG] Weighted Instructions: {weighted_instructions}")

    return response_data

# ---------------- Get Instructions ----------------
def get_edge_instruction(from_node: str, to_node: str) -> List[str]:
    for edge in graph["edges"]:
        if edge["from"] == from_node and edge["to"] == to_node:
            return edge.get("instruction", [])
    return []

def path_to_instructions(path: List[str]) -> List[str]:
    instructions = []
    for i in range(len(path) - 1):
        cur = path[i]
        nxt = path[i + 1]
        step_instr = get_edge_instruction(cur, nxt)
        if step_instr:
            instructions.extend(step_instr)
    return instructions

# ---------------- API Endpoint ----------------
class NavRequest(BaseModel):
    start: str
    end: str

@app.post("/start_dest/")
def start_dest(nav: NavRequest):
    path = dijkstra(adjacency, nav.start, nav.end)

    if not path:
        return {"error": "No path found"}

    first_step = path[0]  # only the next hop
    from_node, to_node, instructions = first_step
    return {
        "from": from_node,
        "to": to_node,
        "instructions": instructions
    }

@app.post("/find_nearest_color/")
def find_nearest_color(current_node: str = Form(...), target_color: str = Form(...)):
    result = dijkstra_color(adjacency, graph["nodes"], current_node, target_color)
    if result is None:
        return {"status": "error", "message": "Could not find a reachable QR of that color."}
    return {
        "status": "ok",
        "nearest_node": result["nearest_node"],
        "distance": result["distance"],
        "path": result["path"]
    }

def dijkstra_color(graph_adj, nodes_data, start, target_color):
    """
    Finds the nearest node with the desired color from start using Dijkstra's algorithm.
    
    graph_adj: adjacency dict like {node: [(neighbor, weight, instructions), ...]}
    nodes_data: graph["nodes"] dict with color info
    start: starting node ID
    target_color: desired color to find
    """
    import heapq

    target_color = target_color.lower()
    visited = set()
    queue = [(0, start, [])]  # (total_distance, current_node, path_edges)

    while queue:
        total_dist, current, path = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)

        # Check if current node has the desired color
        node_color = nodes_data.get(current, {}).get("color", "").lower()
        if node_color == target_color:
            return {
                "nearest_node": current,
                "distance": total_dist,
                "path": path  # list of edges (from, to, instructions)
            }

        for neighbor, weight, instructions in graph_adj.get(current, []):
            if neighbor not in visited:
                heapq.heappush(queue, (
                    total_dist + float(weight),
                    neighbor,
                    path + [(current, neighbor, instructions)]
                ))

    # If we exit the loop, no reachable node of that color
    return None

if __name__ == '__main__':
    print("Starting server...")  # Debugging output
    app.run(debug=True)