import cv2
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
from inference import get_model

# --------------------------------------------------
# PATHS
# --------------------------------------------------
SOURCE_VIDEO_PATH = "foatball/data/test_foatball.mp4"
TARGET_VIDEO_PATH = "/content/annotated_output_video8s_color.mp4"

# --------------------------------------------------
# LOAD inference MODEL
# --------------------------------------------------
model = get_model("football-players-detection-3zvbc/17",api_key="fwUmnyRPCwz4KDuRKJOe")
# football-players-detection-3zvbc/11

# --------------------------------------------------
# COLOR PALETTE FOR CLASSES
# ------------------------------------------------__
CLASS_COLORS = {
0: sv.Color.from_hex('#FFD700'),   # Ball - Gold
1: sv.Color.from_hex("#00FF4C"),   # Goalkeeper - Red
2: sv.Color.from_hex('#00BFFF'),   # Player - DeepSkyBlue
3: sv.Color.from_hex("#DF2F10")    # Referee - LimeGreen
}

# --------------------------------------------------
# ANNOTATORS
# --------------------------------------------------
label_annotator = sv.LabelAnnotator(text_scale=0.35, smart_position=True)

ellipse_annotator = sv.EllipseAnnotator(
thickness=2
)

triangle_annotator = sv.TriangleAnnotator(
color=CLASS_COLORS[0]   # Ball color
)

box_annotator = sv.BoxAnnotator(thickness=2)

# --------------------------------------------------
# TRACKER
# --------------------------------------------------
tracker = sv.ByteTrack()
tracker.reset()

# --------------------------------------------------
# VIDEO IO
# --------------------------------------------------
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

for frame in tqdm(frame_generator, total=video_info.total_frames):

    result = model.infer(frame, conf=0.25)[0]
    detections = sv.Detections.from_inference(result)

    # CLASS SEPARATION
    ball = detections[detections.class_id == 0]
    goalkeeper = detections[detections.class_id == 1]
    players = detections[detections.class_id == 2]
    referees = detections[detections.class_id == 3]

    # DEBUG PRINT
    print(f"Ball: {len(ball)}, GK: {len(goalkeeper)}, Players: {len(players)}, Referee: {len(referees)}")

    # PAD BALL BOX
    if len(ball) > 0:
        ball.xyxy = sv.pad_boxes(ball.xyxy, px=5, py=5)

    # MERGE HUMANS FOR TRACKING
    humans = sv.Detections.merge([players, goalkeeper, referees])
    humans = tracker.update_with_detections(humans)

    # TRACKER LABELS
    labels = [f"ID {tid}" for tid in humans.tracker_id]

    # ANNOTATE FRAME
    annotated_frame = frame.copy()

    # Humans - Ellipse colored by class
    for cls_id in [1,2,3]:
        cls_detections = humans[humans.class_id == cls_id]
        if len(cls_detections) > 0:
            ellipse_annotator.annotate(
                scene=annotated_frame,
                detections=cls_detections
            )

    # Track IDs
    annotated_frame = label_annotator.annotate(annotated_frame, humans, labels)

    # Ball - Triangle
    if len(ball) > 0:
        triangle_annotator.annotate(annotated_frame, ball)

    # Goalkeeper - Box (optional)
    if len(goalkeeper) > 0:
        box_annotator.annotate(
            scene=annotated_frame,
            detections=goalkeeper,
            
        )

    
    cv2.imshow("Annotated Frame", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
#print("✅ Annotated video saved at:", TARGET_VIDEO_PATH)