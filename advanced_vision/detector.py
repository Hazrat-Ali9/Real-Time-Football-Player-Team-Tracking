"""
detector.py - বল এবং খেলোয়াড় ডিটেক্ট করে
"""
import numpy as np
from config import BALL_DISTANCE_THRESHOLD
from utils import box_center, calculate_distance


class BallDetector:
    """বল ডিটেক্ট এবং মালিক নির্ধারণ করে"""
    
    def __init__(self):
        self.ball_threshold = BALL_DISTANCE_THRESHOLD
    
    def get_current_owner(self, detection_results):
        """
        বর্তমান ফ্রেমের বল মালিক নির্ধারণ করে
        
        Args:
            detection_results: ডিটেকশন রেজাল্ট ডিকশনারি
        
        Returns:
            tuple: (owner_id, owner_position) বা (None, None)
        """
        # ডিটেকশন রেজাল্ট থেকে বল এবং খেলোয়াড় আলাদা করুন
        ball_boxes = detection_results.get('ball', [])
        player_boxes = detection_results.get('players', {}).get('boxes', [])
        player_ids = detection_results.get('players', {}).get('ids', [])
        
        # যদি বল না পাওয়া যায় বা খেলোয়াড় না পাওয়া যায়
        if not ball_boxes or not player_boxes:
            return None, None
        
        # বলের প্রথম বক্স নিন
        ball_box = ball_boxes[0]
        ball_center = box_center(ball_box)
        
        # সবচেয়ে কাছের খেলোয়াড় খুঁজুন
        min_distance = float('inf')
        owner_id = None
        owner_position = None
        
        for box, pid in zip(player_boxes, player_ids):
            if pid is None:
                continue
            
            player_center = box_center(box)
            distance = calculate_distance(ball_center, player_center)
            
            # থ্রেশহোল্ডের মধ্যে এবং সবচেয়ে কাছের হলে
            if distance < min_distance and distance < self.ball_threshold:
                min_distance = distance
                owner_id = pid
                owner_position = player_center
        
        return owner_id, owner_position
    
    
    def simulate_detection(self, frame_idx):
        """
        টেস্টিং এর জন্য ডামি ডিটেকশন ডেটা জেনারেট করে
        """
        # ডামি ডেটা - বাস্তবে YOLO/Detectron2 ব্যবহার করবেন
        
        # বল অবস্থান (ডামি)
        ball_x = 300 + 5 * np.sin(frame_idx * 0.1)
        ball_y = 150 + 3 * np.cos(frame_idx * 0.05)
        ball_box = [[ball_x-10, ball_y-10, ball_x+10, ball_y+10]]
        
        # খেলোয়াড় ডেটা (ডামি)
        player_boxes = []
        player_ids = []
        
        # 4 জন খেলোয়াড়
        for i in range(4):
            player_id = 7 if i < 2 else 10  # 2 জন দল A, 2 জন দল B
            offset_x = i * 50
            player_box = [
                ball_x - 30 + offset_x,
                ball_y - 40 + (i % 2) * 30,
                ball_x + 30 + offset_x,
                ball_y + 40 + (i % 2) * 30
            ]
            player_boxes.append(player_box)
            player_ids.append(player_id)
        
        detection_results = {
            'ball': ball_box,
            'players': {
                'boxes': player_boxes,
                'ids': player_ids
            }
        }
        
        return detection_results