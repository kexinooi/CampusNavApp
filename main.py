import heapq
import json
import os
import cv2
import numpy as np
from pyzbar.pyzbar import decode
from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

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

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- QR Constants -----------------
KNOWN_QR_WIDTH_CM = 10
FOCAL_LENGTH = 667

# ----------------- Models -----------------
class NavigationRequest(BaseModel):
    start: str
    end: str
    target_color: str = None

# ----------------- Distance Estimation -----------------
def estimate_distance(bbox_width_px, focal_length_px=498.24):
    KNOWN_QR_WIDTH_CM = 15.0
    MAX_DISTANCE_M = 10.0
    MIN_DISTANCE_M = 0.1
    SAFETY_FACTOR = 1.05
    
    if bbox_width_px <= 0:
        return float('inf')

    distance_m = (KNOWN_QR_WIDTH_CM * focal_length_px) / bbox_width_px / 100.0
    distance_m *= SAFETY_FACTOR
    return max(MIN_DISTANCE_M, min(distance_m, MAX_DISTANCE_M))

def get_color_mask(hsv, color, debug=False):
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
        cv2.imwrite(f"debug_mask_{color}.jpg", mask)
    return mask

def decode_qr_from_frame(frame, target_color=None, debug=False):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if target_color:
        mask = get_color_mask(hsv, target_color.lower())
        if mask is not None:
            frame = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    decoded_objs = decode(gray)

    results = []
    for obj in decoded_objs:
        rect = obj.rect
        results.append({
            "data": obj.data.decode("utf-8"),
            "rect": [rect.left, rect.top, rect.width, rect.height],
            "color": target_color if target_color else "any"
        })
    return results

def dijkstra(graph_adj, start, end):
    queue = [(0, start, [])]
    visited = set()
    while queue:
        cost, node, path_edges = heapq.heappop(queue)
        if node == end:
            return path_edges
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
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"status": "error", "message": "Failed to decode image"}

    response_data = {"status": "waiting", "instructions": [], "qr_data": None, "estimated_distance": None}
    decoded_objs = decode_qr_from_frame(frame, target_color)

    if decoded_objs:
        nearest_obj = max(decoded_objs, key=lambda x: x['rect'][2])
        x, y, w, h = nearest_obj['rect']
        distance_m = estimate_distance(w)
        response_data["estimated_distance"] = distance_m

        if distance_m > 0.5:
            response_data["status"] = "pre_scan"
        else:
            response_data["status"] = "scanned"
            response_data["qr_data"] = nearest_obj['data'].strip().upper()

    return response_data

@app.post("/start_dest/")
def start_dest(nav: NavigationRequest):
    path = dijkstra(adjacency, nav.start, nav.end)
    if not path:
        return {"error": "No path found"}
    from_node, to_node, instructions = path[0]
    return {"from": from_node, "to": to_node, "instructions": instructions}

@app.post("/find_nearest_color/")
def find_nearest_color(current_node: str = Form(...), target_color: str = Form(...)):
    result = dijkstra_color(adjacency, graph["nodes"], current_node, target_color)
    if result is None:
        return {"status": "error", "message": "Could not find a reachable QR of that color."}
    return result

def dijkstra_color(graph_adj, nodes_data, start, target_color):
    target_color = target_color.lower()
    visited = set()
    queue = [(0, start, [])]
    while queue:
        total_dist, current, path = heapq.heappop(queue)
        if current in visited:
            continue
        visited.add(current)
        node_color = nodes_data.get(current, {}).get("color", "").lower()
        if node_color == target_color:
            return {"nearest_node": current, "distance": total_dist, "path": path}
        for neighbor, weight, instructions in graph_adj.get(current, []):
            if neighbor not in visited:
                heapq.heappush(queue, (total_dist + float(weight), neighbor, path + [(current, neighbor, instructions)]))
    return None
