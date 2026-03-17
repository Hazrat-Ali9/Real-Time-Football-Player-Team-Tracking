"""
main.py - প্রধান ফাইল, এইটা চালু করবেন
"""
import cv2
import numpy as np
from config import DEFAULT_FPS, TEAM_A_PLAYERS, TEAM_B_PLAYERS
from detector import BallDetector
from analyzer import PassAnalyzer
from visualizer import FootballVisualizer


def main():
    """প্রধান ফাংশন"""
    print("="*60)
    print("FOOTBALL PASS ANALYSIS SYSTEM")
    print("="*60)
    
    # ভিডিও ফাইল পাথ
    video_path = "notex/0bfacc_0.mp4"  # আপনার ভিডিও ফাইল পাথ
    
    # ভিডিও ক্যাপচার
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        print("Using dummy video feed instead...")
        # ডামি ভিডিও তৈরি (যদি ভিডিও না থাকে)
        cap = create_dummy_video()
    
    # FPS পাওয়া
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = DEFAULT_FPS
    
    print(f"Video FPS: {fps}")
    print(f"Team A Players: {TEAM_A_PLAYERS}")
    print(f"Team B Players: {TEAM_B_PLAYERS}")
    print("-"*60)
    
    # সিস্টেম কম্পোনেন্টস তৈরি
    detector = BallDetector()
    analyzer = PassAnalyzer(fps=fps)
    visualizer = FootballVisualizer()
    
    # প্রধান লুপ
    frame_idx = 0
    print("Starting analysis... Press 'q' to quit")
    
    while True:
        # ফ্রেম পড়া
        ret, frame = cap.read()
        if not ret:
            print("\nVideo ended or cannot read frame.")
            break
        
        # ডিটেকশন (ডামি বা আসল)
        detection_results = detector.simulate_detection(frame_idx)
        
        # বল মালিক নির্ধারণ
        owner_id, owner_position = detector.get_current_owner(detection_results)
        
        # দল নির্ধারণ
        team = 'A' if owner_id in TEAM_A_PLAYERS else 'B' if owner_id in TEAM_B_PLAYERS else None
        
        # পাস ডিটেক্ট
        pass_event = analyzer.detect_pass(owner_id, owner_position, frame_idx)
        
        # পরিসংখ্যান পাওয়া
        current_stats = analyzer.get_stats()
        
        # ভিজ্যুয়ালাইজেশন ---------------------------------
        
        # ১. খেলোয়াড় আঁকা (ডামি - আসলে আপনার ডিটেকশন থেকে আসবে)
        dummy_players = [
            {'id': 7, 'position': (200, 150), 'team': 'A'},
            {'id': 10, 'position': (300, 160), 'team': 'B'},
            {'id': 9, 'position': (250, 200), 'team': 'A'},
            {'id': 11, 'position': (350, 180), 'team': 'B'},
        ]
        
        for player in dummy_players:
            has_ball = (player['id'] == owner_id)
            frame = visualizer.draw_player(
                frame, player['id'], player['position'], 
                player['team'], has_ball
            )
        
        # ২. বল আঁকা (যদি মালিক থাকে)
        if owner_position is not None:
            cv2.circle(frame, 
                      (int(owner_position[0]), int(owner_position[1])), 
                      8, (0, 255, 255), -1)
        
        # ৩. পাস আঁকা
        if pass_event:
            frame = visualizer.draw_pass_arrow(
                frame,
                pass_event['from_position'],
                pass_event['to_position'],
                pass_event['pass_type'],
                pass_event['time']
            )
            
            # সফল পাস হাইলাইট
            if pass_event.get('successful'):
                cv2.putText(frame, "SUCCESSFUL PASS!", 
                           (frame.shape[1]//2 - 100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # ৪. পরিসংখ্যান প্যানেল
        frame = visualizer.draw_stats_panel(frame, current_stats)
        
        # ৫. পাস ইতিহাস
        frame = visualizer.draw_pass_history(frame, analyzer.pass_events)
        
        # ৬. বর্তমান মালিক দেখানো
        if owner_id is not None:
            cv2.putText(frame, f"Ball Owner: Player {owner_id} (Team {team})", 
                       (50, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ৭. ফ্রেম নম্বর দেখানো
        cv2.putText(frame, f"Frame: {frame_idx}", 
                   (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # ৮. ফ্রেম দেখানো
        cv2.imshow('Football Pass Analysis', frame)
        
        # কী প্রেস চেক
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nUser pressed 'q'. Stopping analysis...")
            break
        elif key == ord('p'):
            print("\nPaused. Press any key to continue...")
            cv2.waitKey(0)
        
        frame_idx += 1
        
        # প্রতি 100 ফ্রেমে প্রগ্রেস দেখানো
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")
    
    # রিসোর্স রিলিজ
    cap.release()
    cv2.destroyAllWindows()
    
    # ফাইনাল রিপোর্ট
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    analyzer.print_summary()
    
    print(f"\nTotal frames analyzed: {frame_idx}")
    print(f"Total time: {frame_idx/fps:.2f} seconds")
    print("="*60)


def create_dummy_video():
    """
    ডামি ভিডিও তৈরি করে (যদি ভিডিও ফাইল না থাকে)
    """
    print("Creating dummy video for testing...")
    
    # ডামি ভিডিও ক্যাপচার অবজেক্ট
    class DummyVideoCapture:
        def __init__(self):
            self.width = 640
            self.height = 480
            self.frame_count = 0
            
        def read(self):
            # ডামি ফ্রেম তৈরি
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # কিছু টেক্সট যোগ
            cv2.putText(frame, "DUMMY FOOTBALL VIDEO", 
                       (self.width//2 - 150, self.height//2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.putText(frame, f"Frame: {self.frame_count}", 
                       (self.width//2 - 100, self.height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.putText(frame, "Press 'q' to quit", 
                       (self.width//2 - 100, self.height//2 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            self.frame_count += 1
            return True, frame
        
        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FPS:
                return 30
            elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
                return 900  # 30 সেকেন্ড
            return 0
        
        def isOpened(self):
            return True
        
        def release(self):
            pass
    
    return DummyVideoCapture()


if __name__ == "__main__":
    main()