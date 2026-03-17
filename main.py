import cv2
import os
import numpy as np
from ultralytics import YOLO
import supervision as sv
from tqdm import tqdm
from collections import defaultdict

# =========================================
# PATHS (update these)
# =========================================
VIDEO_PATH = "notex/08fd33_0.mp4"
MODEL_PATH = "foatball/weights/foatball350.pt"
OUTPUT_VIDEO = "outputs/football_analytics_output.mp4"
os.makedirs("outputs", exist_ok=True)

# =========================================
# CONSTANTS
# =========================================
BALL_CLASS = 0
PLAYER_CLASS = 2
POSSESSION_DISTANCE = 60      # pixels
PITCH_WIDTH_METERS = 68       # meters
TOP_N_SPEEDS = 5              # leaderboard

# =========================================
# LOAD MODEL & TRACKER
# =========================================
model = YOLO(MODEL_PATH)
tracker = sv.ByteTrack()

# Dummy team classifier: alternate teams
def team_classifier(player_ids):
    return [pid % 2 for pid in player_ids]  # 0 = Team A, 1 = Team B

# =========================================
# VIDEO SETUP
# =========================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

METERS_PER_PIXEL = PITCH_WIDTH_METERS / width
print(f"🎥 Video Loaded | {width}x{height} | FPS: {fps}")

# =========================================
# STORAGE
# =========================================
player_positions = defaultdict(list)
player_last_position = {}
player_total_distance = defaultdict(float)
player_speeds = defaultdict(list)
player_possession_frames = defaultdict(int)
player_info_hover = {}  # for mouse hover info

# Heatmaps
global_heatmap = np.zeros((height, width), dtype=np.float32)
team_heatmaps = {0: np.zeros((height, width), dtype=np.float32), 1: np.zeros((height, width), dtype=np.float32)}
ball_heatmap = np.zeros((height, width), dtype=np.float32)

last_possessor = None
last_player_boxes = {}

# =========================================
# UTILS
# =========================================
def get_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def overlay_heatmap(frame, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap_blur = cv2.GaussianBlur(heatmap, (0,0), 25)
    heatmap_norm = cv2.normalize(heatmap_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_norm, colormap)
    return cv2.addWeighted(frame, 1-alpha, heatmap_color, alpha, 0)

# =========================================
# MOUSE HOVER CALLBACK
# =========================================
def mouse_hover(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        for pid, box in param.items():
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                info = player_info_hover.get(pid, {})
                print(f"Player {pid} Info: Distance={info.get('distance',0):.2f}m, AvgSpeed={info.get('speed',0):.2f} m/s, Possession={info.get('possession',0):.2f}s")

cv2.namedWindow("Football Analytics Dashboard")
cv2.setMouseCallback("Football Analytics Dashboard", mouse_hover, last_player_boxes)

# =========================================
# MAIN LOOP
# =========================================
frame_generator = sv.get_video_frames_generator(VIDEO_PATH)

for frame_idx, frame in enumerate(tqdm(frame_generator, total=total_frames, desc="Processing")):
    annotated = frame.copy()
    results = model.predict(frame, conf=0.3, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    ball_center = None
    player_centers = {}
    player_ids = []
    last_player_boxes.clear()

    # -------------------------------
    # COLLECT DATA
    # -------------------------------
    for xyxy, cls, track_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
        center = get_center(xyxy)
        player_ids.append(track_id)
        last_player_boxes[track_id] = tuple(map(int, xyxy))

        if cls == BALL_CLASS:
            ball_center = center
            bx, by = int(center[0]), int(center[1])
            if 0 <= bx < width and 0 <= by < height:
                ball_heatmap[by, bx] += 1

        if cls == PLAYER_CLASS:
            player_centers[track_id] = center
            player_positions[track_id].append(center)

            # Update speed & distance
            if track_id in player_last_position:
                prev = player_last_position[track_id]
                pixel_dist = np.linalg.norm(center - prev)
                meter_dist = pixel_dist * METERS_PER_PIXEL
                player_total_distance[track_id] += meter_dist
                player_speeds[track_id].append(meter_dist * fps)
            player_last_position[track_id] = center

            # Update heatmaps
            x, y = int(center[0]), int(center[1])
            global_heatmap[y, x] += 1

            # Update hover info
            avg_speed = np.mean(player_speeds[track_id]) if player_speeds[track_id] else 0
            possession_sec = player_possession_frames[track_id]/fps
            player_info_hover[track_id] = {"distance": player_total_distance[track_id],
                                           "speed": avg_speed,
                                           "possession": possession_sec}

    # -------------------------------
    # TEAM HEATMAPS
    # -------------------------------
    team_ids = team_classifier(player_ids)
    for pid, tid in zip(player_ids, team_ids):
        if pid in player_centers:
            x, y = map(int, player_centers[pid])
            team_heatmaps[tid][y, x] += 1

    # -------------------------------
    # BALL POSSESSION & PASS DETECTION
    # -------------------------------
    possessor = None
    if ball_center is not None:
        min_dist = float("inf")
        for pid, center in player_centers.items():
            dist = np.linalg.norm(ball_center - center)
            if dist < min_dist and dist < POSSESSION_DISTANCE:
                min_dist = dist
                possessor = pid

        if possessor is not None:
            player_possession_frames[possessor] += 1
            if last_possessor is not None and last_possessor != possessor:
                start = tuple(player_centers[last_possessor].astype(int))
                end = tuple(player_centers[possessor].astype(int))
                cv2.line(annotated, start, end, (0,255,255), 2)
            last_possessor = possessor

    # -------------------------------
    # DRAW PLAYERS & BALL
    # -------------------------------
    for xyxy, cls, track_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
        x1, y1, x2, y2 = map(int, xyxy)
        if cls == BALL_CLASS:
            color = (0,255,255)
        elif cls == PLAYER_CLASS:
            color = (0,0,255) if team_classifier([track_id])[0]==0 else (0,255,0)
        else:
            color = (255,255,255)
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated, f"ID:{track_id}", (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.putText(annotated, f"Frame: {frame_idx}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # -------------------------------
    # OVERLAY HEATMAPS
    # -------------------------------
    annotated = overlay_heatmap(annotated, global_heatmap, alpha=0.3, colormap=cv2.COLORMAP_JET)
    team_overlay = cv2.addWeighted(
        overlay_heatmap(annotated, team_heatmaps[0], alpha=0.3, colormap=cv2.COLORMAP_JET),
        0.5,
        overlay_heatmap(annotated, team_heatmaps[1], alpha=0.3, colormap=cv2.COLORMAP_HOT),
        0.5, 0
    )
    annotated = cv2.addWeighted(annotated, 0.7, team_overlay, 0.3, 0)
    annotated = overlay_heatmap(annotated, ball_heatmap, alpha=0.5, colormap=cv2.COLORMAP_OCEAN)

    # -------------------------------
    # TOP SPEED LEADERBOARD
    # -------------------------------
    top_speeds = sorted([(pid, np.mean(speeds)) for pid, speeds in player_speeds.items() if speeds], key=lambda x:x[1], reverse=True)[:TOP_N_SPEEDS]
    y_offset = 50
    for rank, (pid, speed) in enumerate(top_speeds, 1):
        cv2.putText(annotated, f"{rank}. Player {pid}: {speed:.2f} m/s", (width-300, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_offset += 30

    # -------------------------------
    # WRITE & DISPLAY
    # -------------------------------
    out.write(annotated)
    cv2.imshow("Football Analytics Dashboard", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
out.release()
print("✅ Output video saved:", OUTPUT_VIDEO)

# =========================================
# OFFLINE HEATMAP IMAGES
# =========================================
HEATMAP_DIR = "outputs/heatmaps"
os.makedirs(HEATMAP_DIR, exist_ok=True)

for pid, positions in player_positions.items():
    heatmap = np.zeros((height,width), dtype=np.float32)
    for x, y in positions:
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += 1
    heatmap = cv2.GaussianBlur(heatmap, (0,0), 25)
    heatmap_color = cv2.applyColorMap(cv2.normalize(heatmap,None,0,255,cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(f"{HEATMAP_DIR}/player_{pid}.jpg", heatmap_color)

print("🔥 Heatmaps saved in", HEATMAP_DIR)

# =========================================
# FINAL PLAYER STATS
# =========================================
print("\n📊 PLAYER STATS")
for pid in player_total_distance:
    possession_sec = player_possession_frames[pid] / fps
    avg_speed = np.mean(player_speeds[pid]) if player_speeds[pid] else 0
    print(f"Player {pid} | Distance: {player_total_distance[pid]:.2f} m | Avg Speed: {avg_speed:.2f} m/s | Possession: {possession_sec:.2f} s")
