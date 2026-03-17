
import os
import cv2
import numpy as np
from tqdm import tqdm

import supervision as sv
from ultralytics import YOLO
from sports.common.team import TeamClassifier
from collections import defaultdict


from foatball.config import *
from foatball.pass_tracker import PassTracker
from foatball.visualizer import FootballVisualizer
from foatball.utilitis.utils import *
from foatball.utilitis import TeamClassifier,extrack_player_crops

# ============================================
# CONFIG
# ============================================
SOURCE_VIDEO_PATH = "foatball/data/121364_0.mp4"          # change
OUTPUT_VIDEO_PATH = "output_with_passes.mp4"
MODEL_PATH = "foatball/weights/foatball350.pt"             # change

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
CONF_THRES = 0.5

BALL_CLASS = 0
PLAYER_CLASS = 2
POSSESSION_DISTANCE = 60
POSSESSION_RADIUS = 40  # pixel (tune it)
PITCH_WIDTH_METERS = 68
TOP_N_SPEEDS = 4
player_positions = defaultdict(list)
player_last_position = {}
player_total_distance = defaultdict(float)
player_speeds = defaultdict(list)
player_possession_frames = defaultdict(int)
player_possession_time = defaultdict(float)


# ============================================
# INITIALIZE SYSTEMS
# ============================================
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# পাস ট্র্যাকার তৈরি
pass_tracker = PassTracker(fps=30)
visualizer = FootballVisualizer(pass_tracker)

# ============================================
# EXTRACT PLAYER CROPS (TEAM TRAINING)
# ============================================
def extract_player_crops(video_path, stride=20, max_crops=120):
    print("[INFO] Extracting player crops...")

    crops = []
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % stride != 0:
            frame_idx += 1
            continue

        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        result = model.track(frame, persist=True, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(result)

        # class 2 = player
        players = detections[detections.class_id == 2]

        for box in players.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

        if len(crops) >= max_crops:
            break

        frame_idx += 1

    cap.release()
    print(f"[INFO] Collected {len(crops)} player crops")
    return crops

def get_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2)/2, (y1 + y2)/2])

# ============================================
# TRAIN TEAM CLASSIFIER
# ============================================
# player_crops = extract_player_crops(SOURCE_VIDEO_PATH)

# print("[INFO] Training Team Classifier...")
# team_classifier = TeamClassifier(device="cuda" if cv2.cuda.getCudaEnabledDeviceCount() else "cpu")
# team_classifier.fit(player_crops)
# print("[INFO] Team Classifier ready!")

# ============================================
# ANNOTATORS
# ============================================
team_palette = sv.ColorPalette.from_hex(["#14FFF7"])

ellipse_annotator = sv.EllipseAnnotator(color=team_palette, thickness=2)
triangle_ball = sv.TriangleAnnotator(color=sv.Color.YELLOW)
box_ref = sv.BoxAnnotator(color=sv.Color.RED)
box_gk = sv.BoxAnnotator(color=sv.Color.GREEN)

label_player = sv.LabelAnnotator(
    color=team_palette,
    #color_lookup = sv.ColorLookup.CLASS,
    text_color=sv.Color.BLACK,
    text_scale=0.5,text_padding = 0
)

label_ref = sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.4,text_padding = 5)
label_gk = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=0.4)

tracker = sv.ByteTrack()

# ============================================
# VIDEO WRITER
# ============================================
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    fps,
    (TARGET_WIDTH, TARGET_HEIGHT)
)
METERS_PER_PIXEL = PITCH_WIDTH_METERS / width


# ============================================
# MAIN LOOP WITH PASS TRACKING
# ============================================
print("[INFO] Processing video with pass tracking...")
frame_gen = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# পাস ইভেন্ট স্টোর করতে
all_pass_events = []

for frame_idx, frame in enumerate(tqdm(frame_gen)):
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    annotated = frame.copy()
    
    try:
        result = model.track(frame, persist=True, conf=CONF_THRES)[0]
        detections = sv.Detections.from_ultralytics(result)

        ball = detections[detections.class_id == 0]
        goalkeeper = detections[detections.class_id == 1]
        players = detections[detections.class_id == 2]
        referee = detections[detections.class_id == 3]

        # NMS + Tracking
        players = players.with_nms(0.3)
        players = tracker.update_with_detections(players)
        #print("players Deatils :",players)

        # ===== TEAM CLASSIFICATION =====
        crops, valid_ids = [], []
        ball_center = None
        player_centers = {}
        player_ids = []
        nearest_player = None
        min_dist = float("inf")

        for i, (box, cls, track_id) in enumerate(zip(players.xyxy, players.class_id, players.tracker_id)):

            x1, y1, x2, y2 = map(int, box)
            center = get_center(box)

            player_ids.append(track_id)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                valid_ids.append(i)

            if cls == PLAYER_CLASS:
                player_centers[track_id] = center
                player_positions[track_id].append(center)

                if track_id in player_last_position:
                    prev = player_last_position[track_id]
                    pixel_dist = np.linalg.norm(center - prev)
                    meter_dist = pixel_dist * METERS_PER_PIXEL

                    player_total_distance[track_id] += meter_dist
                    player_speeds[track_id].append(meter_dist * fps)

                player_last_position[track_id] = center

            # Detect Ball
            if cls == BALL_CLASS:
                ball_center = center

            # Player logic
            if cls == PLAYER_CLASS and ball_center is not None:
                dist_to_ball = np.linalg.norm(center - ball_center)

                if dist_to_ball < min_dist:
                    min_dist = dist_to_ball
                    nearest_player = track_id
                    nearest_player.append(track_id)
            
            # Assign possession
            if nearest_player is not None and min_dist < POSSESSION_RADIUS:
                player_possession_time[nearest_player] += 1 / fps

        # team_ids = [0] * len(players)
        # if crops:
        #     preds = team_classifier.predict(crops)
        #     for idx, team_id in zip(valid_ids, preds):
        #         team_ids[idx] = team_id

        # players.class_id = team_ids

        # ===== BALL OWNER DETECTION =====
        owner_id, owner_position, owner_team = pass_tracker.find_ball_owner(
            ball, players, frame_idx
        )
        
        current_time = frame_idx / fps
        
        # ===== PASS DETECTION =====
        pass_event = pass_tracker.update_possession(
            owner_id, owner_position, owner_team, frame_idx, current_time
        )
        #print("Pass_event:",pass_event)
        if pass_event:
            all_pass_events.append(pass_event)
        
        # ===== GET CURRENT STATS =====
        current_stats = pass_tracker.get_current_stats()
        #print("current_stats:",current_stats)
        
        # ===== LABELS =====
        labels = []
        for tid, pid in zip(players.class_id, players.tracker_id):
            labels.append(f"T{tid+1}-{pid}")

        # ===== DRAW ORIGINAL ANNOTATIONS =====
        if len(players):
            annotated = ellipse_annotator.annotate(annotated, players)
            annotated = label_player.annotate(annotated, players, labels)

        if len(ball):
            annotated = triangle_ball.annotate(annotated, ball)

        if len(goalkeeper):
            annotated = box_gk.annotate(annotated, goalkeeper)
            annotated = label_gk.annotate(
                annotated, goalkeeper,
                [f"GK {c:.2f}" for c in goalkeeper.confidence]
            )

        if len(referee):
            annotated = box_ref.annotate(annotated, referee)
            annotated = label_ref.annotate(
                annotated, referee,
                [f"REF {c:.2f}" for c in referee.confidence]
            )
        
        # ===== DRAW PASS VISUALIZATIONS =====
        
        # ১. বল মালিক দেখানো
        if owner_id is not None:
            annotated = visualizer.draw_ball_owner(
                annotated, owner_id, owner_position, owner_team
            )
        
        # ২. পাস তীর আঁকা (যদি পাস হয়)
        if pass_event and SHOW_PASS_ARROWS:
            annotated = visualizer.draw_pass_arrow(annotated, pass_event)
        
        # ৩. পরিসংখ্যান প্যানেল
        if SHOW_POSSESSION_STATS:
            annotated = visualizer.draw_stats_panel(annotated, current_stats)
        
        # ৪. পাস ইতিহাস
        if SHOW_PASS_HISTORY:
            annotated = visualizer.draw_pass_history(
                annotated, current_stats.get('recent_passes', [])
            )
        
        # ৫. ফ্রেম তথ্য
        annotated = visualizer.draw_frame_info(annotated, frame_idx, fps)
        # Top Speed Leaderboard
        top_speeds = sorted([(pid, np.mean(speeds)) for pid, speeds in player_speeds.items() if speeds], key=lambda x:x[1], reverse=True)[:TOP_N_SPEEDS]
        y_offset = 610
        for rank, (pid, speed) in enumerate(top_speeds, 1):
            cv2.putText(annotated, f"{rank}. Player {pid}: {speed:.2f} m/s", (width-920, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 0, 100), 2)
            y_offset += 30

        if nearest_player is not None:
            cv2.putText(frame,f"Ball Possession: Player {nearest_player}",(frame.shape[1] - 520, 70),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)

        
        # ৬. ফ্রেম নম্বর
        # cv2.putText(
        #     annotated, f"Frame {frame_idx}",
        #     (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7, (255, 255, 255), 2
        # )

    except Exception as e:
        print(f"[WARN] Frame {frame_idx} error: {e}")
        annotated = frame

    out.write(annotated)
    cv2.imshow("Football Analytics Dashboard", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
out.release()

# ============================================
# FINALIZE
# ============================================


print("\n" + "="*60)
print("PROCESSING COMPLETE!")
print("="*60)

# ফাইনাল রিপোর্ট
pass_tracker.print_summary()

print(f"\n📊 Total Pass Events Detected: {len(all_pass_events)}")
print(f"📁 Output saved to: {OUTPUT_VIDEO_PATH}")

# পাস ডেটা সেভ (ঐচ্ছিক)
if all_pass_events:
    import json
    with open('pass_events.json', 'w') as f:
        json.dump(all_pass_events, f, indent=2)
    print("💾 Pass events saved to pass_events.json")

print("="*60)