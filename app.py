import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO
from sports.common.team import TeamClassifier
from foatball.utilitis import extrack_player_crops, TeamClassifier
import cv2


SOURCE_VIDEO_PATH = "notex/573e61_0.mp4"
TARGET_VIDEO_PATH = "annotated_output_video1.mp4"
model = YOLO("foatball/weights/foatball350.pt")  


# EXTRACT PLAYER CROPS FUNCTION
def extrack_player_crops(source,stride):
    frame_generator = sv.get_video_frames_generator(source,stride=stride)
    crops = []
    for frame in tqdm(frame_generator):  # process only first 100 frames for demo
        results = model.track(frame, persist=True,conf=0.20)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections.with_nms(threshold=0.3,class_agnostic=True)
        detections = detections[ detections.class_id == 2 ]  # keep only class_id 0 (players)
        crops += [
            sv.crop_image(frame,xyxy)
            for xyxy in detections.xyxy
            ]
        print(f"Cropped {len(crops)} player images from frame.")
    return crops


# --------------------------------------------------
# TRAIN TEAM CLASSIFIER
crops = extrack_player_crops(SOURCE_VIDEO_PATH,20)
team_classifier = TeamClassifier()
team_classifier.fit(crops)
print(f"[INFO] Training crops: {len(crops)}")



# --------------------------------------------------
# ANNOTATORS
box_goalkeeper = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=2)
box_referee = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)

ellipse_player = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
                                    thickness=2)
triangle_ball = sv.TriangleAnnotator(color=sv.Color.YELLOW,base=25,height=20)

label_player = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER,
    text_scale=0.5,
    text_thickness=0,text_padding=3,
    smart_position=True
)
label_referee = sv.LabelAnnotator(color=sv.Color.RED, text_scale=0.3, smart_position=True)
label_goalkeeper = sv.LabelAnnotator(color=sv.Color.GREEN, text_scale=0.3, smart_position=True)


# --------------------------------------------------
# TRACKER
tracker = sv.ByteTrack()
tracker.reset()


video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)

frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
#frame = next(frame_generator)

with video_sink:
    for frame in frame_generator:
        frame = cv2.resize(frame, (1280,720))
        result = model.track(frame,persist=True, conf=0.5)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Ball and player tracking and annotation
        # {0: 'ball', 1: 'goalkeeper', 2: 'player', 3: 'referee'}
        ball_detection = detections[detections.class_id == 0]
        ball_detection.xyxy = sv.pad_boxes(ball_detection.xyxy,px=5,py=5)

        player_detection = detections[detections.class_id != 0]
        #player_detection.class_id -= 1
        player_detection = player_detection.with_nms(threshold=0.3,class_agnostic=True)

        goalkeeper_detection = detections[detections.class_id ==1]
        referee_detection = detections[detections.class_id ==3]

        # --------------------------------------------------
        # TRACK FIRST, THEN TEAM CLASSIFY
        player_detection = tracker.update_with_detections(player_detection)
        player_crops = [sv.crop_image(frame, xyxy)for xyxy in player_detection.xyxy]

        team_ids = team_classifier.predict(player_crops)
        player_detection.class_id = team_ids   
        print("Team IDs:", set(team_ids))


        # labels = [
        #     f"{class_name} {confidence:.2f}"
        #     for class_name, confidence
        #     in zip(player_detection.data['class_name'], player_detection.confidence)
        # ]
        labels = [
            f"#{tid}"
            for tid in player_detection.tracker_id
        ]

        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = ellipse_player.annotate(
            scene=annotated_frame,
            detections=player_detection
        )
        annotated_frame = triangle_ball.annotate(
            scene=annotated_frame,
            detections=ball_detection,
        )
        annotated_frame = box_goalkeeper.annotate(
            scene=annotated_frame,
            detections=goalkeeper_detection
        )
        annotated_frame = box_referee.annotate(
            scene=annotated_frame,
            detections=referee_detection
        )
        annotated_frame = label_player.annotate(
            scene=annotated_frame,
            detections=player_detection.tracker_id is not None and player_detection,
            labels=labels
        )
        annotated_frame = label_referee.annotate(
            scene=annotated_frame,
            detections=referee_detection,
            labels=[f"Referee {conf:.2f}" for conf in referee_detection.confidence]
        )
        annotated_frame = label_goalkeeper.annotate(
            scene=annotated_frame,
            detections=goalkeeper_detection,
            labels=[f"Goalkeeper {conf:.2f}" for conf in goalkeeper_detection.confidence]
        )
        # annotated_frame = triangle_ball.annotate(
        #     scene=annotated_frame,
        #     detections=ball_detection,
        # )

        #sv.write_video_frame(TARGET_VIDEO_PATH, annotated_frame)
        video_sink.write_frame(annotated_frame)

        cv2.imshow("Annotated Frame", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
