# ==========================================================
# INSTALL DEPENDENCIES (Colab Specific)
# ==========================================================
# !pip install ultralytics supervision opencv-python-headless tqdm

# ==========================================================
# IMPORTS
# ==========================================================
import os
import cv2
import numpy as np
from tqdm import tqdm
from google.colab.patches import cv2_imshow  # Colab specific
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

# Note: For Colab, we'll create mock implementations of missing modules
# You'll need to upload your actual modules or install them

# ==========================================================
# MOCK MODULES FOR MISSING IMPORTS
# ==========================================================
# Create mock implementations for missing modules
print("[INFO] Creating mock implementations for missing modules...")



from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)

class PassTracker:
    def __init__(self, fps=30):
        self.fps = fps
        self.passes = []
        self.current_owner = None
        self.last_owner_change_frame = 0
        print(f"[MOCK] PassTracker initialized with FPS: {fps}")
    
    def find_ball_owner(self, ball_detections, player_detections, frame_idx):
        if len(ball_detections) == 0 or len(player_detections) == 0:
            return None, None, None
        
        # Simple logic: find player closest to ball
        ball_center = np.array([(ball_detections.xyxy[0][0] + ball_detections.xyxy[0][2]) / 2,
                               (ball_detections.xyxy[0][1] + ball_detections.xyxy[0][3]) / 2])
        
        min_dist = float('inf')
        owner_id = None
        owner_position = None
        owner_team = None
        
        for i, (box, track_id) in enumerate(zip(player_detections.xyxy, player_detections.tracker_id)):
            player_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            dist = np.linalg.norm(player_center - ball_center)
            
            if dist < min_dist:
                min_dist = dist
                owner_id = track_id
                owner_position = player_center
                owner_team = player_detections.class_id[i] if i < len(player_detections.class_id) else 0
        
        return owner_id, owner_position, owner_team
    
    def update_possession(self, owner_id, owner_position, owner_team, frame_idx, current_time):
        if owner_id != self.current_owner and owner_id is not None:
            if self.current_owner is not None:
                # This is a pass
                pass_event = {
                    'from_player': self.current_owner,
                    'to_player': owner_id,
                    'frame': frame_idx,
                    'time': current_time,
                    'from_team': 0,  # Mock
                    'to_team': owner_team if owner_team is not None else 0
                }
                self.passes.append(pass_event)
                self.current_owner = owner_id
                self.last_owner_change_frame = frame_idx
                return pass_event
            else:
                self.current_owner = owner_id
                self.last_owner_change_frame = frame_idx
        return None
    
    def get_current_stats(self):
        return {
            'total_passes': len(self.passes),
            'recent_passes': self.passes[-5:] if len(self.passes) > 0 else [],
            'current_owner': self.current_owner,
            'possession_time': (len(self.passes) * 2)  # Mock value
        }
    
    def print_summary(self):
        print("\n" + "="*40)
        print("PASS TRACKER SUMMARY")
        print("="*40)
        print(f"Total passes detected: {len(self.passes)}")
        for i, pass_event in enumerate(self.passes[-5:]):  # Show last 5 passes
            print(f"Pass {i+1}: Player {pass_event['from_player']} -> Player {pass_event['to_player']}")

class FootballVisualizer:
    def __init__(self, pass_tracker):
        self.pass_tracker = pass_tracker
        self.pass_history = []
    
    def draw_ball_owner(self, frame, owner_id, owner_position, owner_team):
        if owner_position is not None:
            # Draw circle around ball owner
            center = tuple(map(int, owner_position))
            cv2.circle(frame, center, 15, (0, 255, 255), 3)
            cv2.putText(frame, f"Owner: P{owner_id}", 
                       (center[0] - 30, center[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

    def draw_stats_panel_details(self, frame, stats):
        """
        পরিসংখ্যান প্যানেল আঁকে
        """
        height, width = frame.shape[:2]
        
        # আধা-স্বচ্ছ প্যানেল
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # পরিসংখ্যান টেক্সট
        stats_text = [
            "⚽ LIVE PASS STATS",
            f"Team A: {stats.get('team_a_possession', '0%')}",
            f"Team B: {stats.get('team_b_possession', '0%')}",
            f"Total Passes: {stats.get('total_passes', 0)}",
            f"Successful: {stats.get('successful_passes', 0)}",
            f"Success Rate: {stats.get('success_rate', '0%')}",
            f"Avg Pass Time: {stats.get('avg_pass_time', '0s')}",
            f"Owner: P{stats.get('current_owner', 'None')}"
        ]
        
        # টেক্সট আঁকা
        y_offset = 40
        line_height = 22
        
        for i, text in enumerate(stats_text):
            color = (0, 255, 155) if i == 0 else (200, 200, 200)
            font_scale = 0.6 if i == 0 else 0.5
            
            cv2.putText(
                frame, text, (20, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1
            )
        return frame
    
    def draw_pass_arrow(self, frame, pass_event):
        # For now, just display pass info
        cv2.putText(frame, f"Pass: {pass_event['from_player']}->{pass_event['to_player']}", 
                   (20, TARGET_HEIGHT - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 0, 100), 2)
        return frame
    
    def draw_stats_panel(self, frame, stats):
        # Draw stats panel
        panel_y = 20
        cv2.putText(frame, f"Total Passes: {stats.get('total_passes', 0)}", 
                   (TARGET_WIDTH - 250, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Current Owner: {stats.get('current_owner', 'None')}", 
                   (TARGET_WIDTH - 250, panel_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame
    
    def draw_pass_history(self, frame, recent_passes):
        # Draw recent passes
        if recent_passes:
            for i, pass_event in enumerate(recent_passes[-3:]):  # Last 3 passes
                cv2.putText(frame, f"Pass {i+1}: P{pass_event['from_player']}->P{pass_event['to_player']}", 
                           (20, TARGET_HEIGHT - 100 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 200), 2)
        return frame
    
    def draw_frame_info(self, frame, frame_idx, fps):
        cv2.putText(frame, f"Frame: {frame_idx}", 
                   (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {frame_idx/fps:.1f}s", 
                   (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame

# ============================================
# CONFIG
# ============================================
SOURCE_VIDEO_PATH = "/content/121364_0.mp4"  # You'll need to upload this
OUTPUT_VIDEO_PATH = "/content/output_with_passes.mp4"
MODEL_PATH = "/content/foatball350.pt"  # You'll need to upload this

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
CONF_THRES = 0.5

BALL_CLASS = 0
PLAYER_CLASS = 2
POSSESSION_DISTANCE = 60
POSSESSION_RADIUS = 40
PITCH_WIDTH_METERS = 68
TOP_N_SPEEDS = 4

# Visualization flags
SHOW_PASS_ARROWS = True
SHOW_POSSESSION_STATS = True
SHOW_PASS_HISTORY = True

# ==========================================================
# STORAGE
# ==========================================================
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

# Check if CUDA is available
import torch
if torch.cuda.is_available():
    print(f"[INFO] CUDA is available. Device: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("[INFO] CUDA not available, using CPU")
    device = "cpu"

# Load model (will fail if file doesn't exist - you need to upload it)
try:
    model = YOLO(MODEL_PATH)
    model.to(device)
    print(f"[INFO] Model loaded successfully on {device}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("[INFO] Using default YOLOv8 model instead")
    model = YOLO("yolov8n.pt")  # Fallback to default model

# Initialize systems
pass_tracker = PassTracker(fps=30)
visualizer = FootballVisualizer(pass_tracker)

# ============================================
# HELPER FUNCTIONS
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
        
        # class 2 = player (in COCO format)
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
# UPLOAD FILES TO COLAB
# ============================================
print("\n" + "="*60)
print("FILE UPLOAD INSTRUCTIONS")
print("="*60)
print("1. Upload your video file to /content/121364_0.mp4")
print("2. Upload your model to /content/foatball350.pt")
print("3. Or use the code below to upload files:")
print("\n   from google.colab import files")
print("   uploaded = files.upload()")
print("="*60)

# Uncomment to enable file upload
# from google.colab import files
# uploaded = files.upload()

# ============================================
# TRAIN TEAM CLASSIFIER
# ============================================
print("\n[INFO] Training Team Classifier...")
try:
    player_crops = extract_player_crops(SOURCE_VIDEO_PATH)
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(player_crops)
    print("[INFO] Team Classifier ready!")
except Exception as e:
    print(f"[WARNING] Could not train team classifier: {e}")
    print("[INFO] Using mock team classifier instead")
    team_classifier = TeamClassifier(device=device)

# ============================================
# ANNOTATORS
# ============================================
team_palette = color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700'])
ellipse_annotator = sv.EllipseAnnotator(color=team_palette, thickness=2)
triangle_ball = sv.TriangleAnnotator(color=sv.Color.YELLOW)
box_ref = sv.BoxAnnotator(color=sv.Color.RED)
box_gk = sv.BoxAnnotator(color=sv.Color.GREEN)

label_player = sv.LabelAnnotator(
    color=team_palette,
    text_color=sv.Color.BLACK,
    text_scale=0.5,
    text_padding=0
)

label_ref = sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.4, text_padding=5)
label_gk = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=0.4)

tracker = sv.ByteTrack()

# ============================================
# VIDEO WRITER
# ============================================
print("[INFO] Setting up video writer...")
cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Could not open video: {SOURCE_VIDEO_PATH}")
    print("[INFO] Creating a dummy video for testing")
    # Create a dummy video for testing
    dummy_frame = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "TEST VIDEO - Upload your video", 
               (100, TARGET_HEIGHT//2), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Save dummy frame as video
    dummy_out = cv2.VideoWriter(SOURCE_VIDEO_PATH, 
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               30, (TARGET_WIDTH, TARGET_HEIGHT))
    for _ in range(100):  # 100 frames
        dummy_out.write(dummy_frame)
    dummy_out.release()
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS) or 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"[INFO] Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc,
    fps,
    (TARGET_WIDTH, TARGET_HEIGHT)
)
METERS_PER_PIXEL = PITCH_WIDTH_METERS / width if width > 0 else 0.05

# ============================================
# MAIN LOOP WITH PASS TRACKING
# ============================================
print(f"[INFO] Processing video with pass tracking...")
print(f"[INFO] Processing {total_frames} frames")

frame_gen = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
all_pass_events = []

# Main processing loop
for frame_idx, frame in enumerate(tqdm(frame_gen, total=total_frames, desc="Processing frames")):
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
    annotated = frame.copy()
    
    try:
        # Run YOLO inference
        result = model.track(frame, persist=True, conf=CONF_THRES)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter detections (adjust class IDs based on your model)
        ball = detections[detections.class_id == BALL_CLASS]
        goalkeeper = detections[detections.class_id == 1] if 1 in detections.class_id else detections[detections.class_id == 0]
        players = detections[detections.class_id == PLAYER_CLASS]
        referee = detections[detections.class_id == 3] if 3 in detections.class_id else sv.Detections.empty()
        
        # Apply NMS and tracking to players
        if len(players):
            players = players.with_nms(0.3)
            players = tracker.update_with_detections(players)
        
        # ===== TEAM CLASSIFICATION - FIXED =====
        crops = []
        valid_indices = []
        ball_center = None
        player_centers = {}
        nearest_player = None
        min_dist = float("inf")
        
        # First pass: collect all player data
        for i, (box, cls, track_id) in enumerate(zip(players.xyxy, players.class_id, players.tracker_id)):
            x1, y1, x2, y2 = map(int, box)
            center = get_center(box)
            
            # Store player center
            player_centers[track_id] = center
            player_positions[track_id].append(center)
            
            # Calculate distance and speed
            if track_id in player_last_position:
                prev = player_last_position[track_id]
                pixel_dist = np.linalg.norm(center - prev)
                meter_dist = pixel_dist * METERS_PER_PIXEL
                player_total_distance[track_id] += meter_dist
                player_speeds[track_id].append(meter_dist * fps)
            
            player_last_position[track_id] = center
            
            # Collect crop for team classification
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                valid_indices.append(i)
        
        # Get ball center if available
        if len(ball):
            ball_center = get_center(ball.xyxy[0])
            
            # Find nearest player to ball
            for track_id, center in player_centers.items():
                dist_to_ball = np.linalg.norm(center - ball_center)
                if dist_to_ball < min_dist:
                    min_dist = dist_to_ball
                    nearest_player = track_id  # FIXED: No .append() here!
        
        # Assign possession time
        if nearest_player is not None and min_dist < POSSESSION_RADIUS:
            player_possession_time[nearest_player] += 1 / fps
        
        # Team classification
        team_ids = [0] * len(players)  # Default all to team 0
        
        if crops and valid_indices:
            try:
                # Get predictions for all crops at once
                predictions = team_classifier.predict(crops)
                print('===========Predictions=============',predictions)
                
                # Assign predictions to the correct players
                for idx, team_id in zip(valid_indices, predictions):
                    if idx < len(team_ids):
                        team_ids[idx] = team_id
            except Exception as e:
                print(f"[WARN] Team classification failed: {e}")
                # Fallback: assign teams alternately
                for i in range(len(team_ids)):
                    team_ids[i] = i % 2
        
        # Update player detections with team IDs
        if len(players):
            players.class_id = np.array(team_ids)
        
        # ===== BALL OWNER DETECTION =====
        owner_id, owner_position, owner_team = pass_tracker.find_ball_owner(
            ball, players, frame_idx
        )
        
        current_time = frame_idx / fps
        
        # ===== PASS DETECTION =====
        pass_event = pass_tracker.update_possession(
            owner_id, owner_position, owner_team, frame_idx, current_time
        )
        
        if pass_event:
            all_pass_events.append(pass_event)
        
        # ===== GET CURRENT STATS =====
        current_stats = pass_tracker.get_current_stats()
        
        # ===== LABELS =====
        labels = []
        if len(players):
            for team_id, track_id in zip(players.class_id, players.tracker_id):
                labels.append(f"T{team_id+1}-{track_id}")
        
        # ===== DRAW ORIGINAL ANNOTATIONS =====
        if len(players):
            annotated = ellipse_annotator.annotate(annotated, players)
            annotated = label_player.annotate(annotated, players, labels)
        
        if len(ball):
            annotated = triangle_ball.annotate(annotated, ball)
        
        if len(goalkeeper):
            annotated = box_gk.annotate(annotated, goalkeeper)
            gk_labels = [f"GK {c:.2f}" for c in goalkeeper.confidence] if hasattr(goalkeeper, 'confidence') else ["GK"]
            annotated = label_gk.annotate(annotated, goalkeeper, gk_labels)
        
        if len(referee):
            annotated = box_ref.annotate(annotated, referee)
            ref_labels = [f"REF {c:.2f}" for c in referee.confidence] if hasattr(referee, 'confidence') else ["REF"]
            annotated = label_ref.annotate(annotated, referee, ref_labels)
        
        # ===== DRAW PASS VISUALIZATIONS =====
        if owner_id is not None:
            annotated = visualizer.draw_ball_owner(
                annotated, owner_id, owner_position, owner_team
            )
        
        if pass_event and SHOW_PASS_ARROWS:
            annotated = visualizer.draw_pass_arrow(annotated, pass_event)
        
        if SHOW_POSSESSION_STATS:
            annotated = visualizer.draw_stats_panel(annotated, current_stats)
        
        if SHOW_PASS_HISTORY:
            annotated = visualizer.draw_pass_history(
                annotated, current_stats.get('recent_passes', [])
            )
        
        # ===== FRAME INFO =====
        annotated = visualizer.draw_frame_info(annotated, frame_idx, fps)
        
        # ===== TOP SPEED LEADERBOARD =====
        if player_speeds:
            avg_speeds = []
            for pid, speeds in player_speeds.items():
                if speeds:
                    avg_speed = np.mean(speeds[-10:]) if len(speeds) > 10 else np.mean(speeds)
                    avg_speeds.append((pid, avg_speed))
            
            avg_speeds.sort(key=lambda x: x[1], reverse=True)
            top_speeds = avg_speeds[:TOP_N_SPEEDS]
            
            y_offset = 610
            for rank, (pid, speed) in enumerate(top_speeds, 1):
                cv2.putText(annotated, f"{rank}. P{pid}: {speed:.2f} m/s", 
                           (TARGET_WIDTH - 320, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 2)
                y_offset += 25
        
        # ===== BALL POSSESSION DISPLAY =====
        if nearest_player is not None:
            cv2.putText(annotated, f"Ball Possession: Player {nearest_player}",
                       (TARGET_WIDTH - 320, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    except Exception as e:
        print(f"\n[WARN] Frame {frame_idx} error: {e}")
        annotated = frame
    
    # Write frame to output video
    out.write(annotated)
    
    # Show preview every 100 frames
    if frame_idx % 100 == 0:
        preview = cv2.resize(annotated, (640, 360))
        cv2_imshow(preview)

# ============================================
# CLEANUP AND FINALIZATION
# ============================================
out.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("PROCESSING COMPLETE!")
print("="*60)

# Final report
pass_tracker.print_summary()

print(f"\n📊 Total Pass Events Detected: {len(all_pass_events)}")
print(f"📁 Output saved to: {OUTPUT_VIDEO_PATH}")

# Print player statistics
if player_total_distance:
    print("\n🏃 Player Distance Summary (Top 10):")
    for pid, dist in sorted(player_total_distance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  Player {pid}: {dist:.2f} meters")

if player_possession_time:
    print("\n⚽ Ball Possession Summary (Top 5):")
    for pid, time in sorted(player_possession_time.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Player {pid}: {time:.2f} seconds")

# # Save pass events
# if all_pass_events:
#     import json
#     with open('/content/pass_events.json', 'w') as f:
#         json.dump(all_pass_events, f, indent=2)
#     print("\n💾 Pass events saved to /content/pass_events.json")

print("="*60)
print("\n✅ All done! Download your output video:")
print(f"   from google.colab import files")
print(f"   files.download('{OUTPUT_VIDEO_PATH}')")