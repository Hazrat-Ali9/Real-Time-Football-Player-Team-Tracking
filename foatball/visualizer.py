"""
visualizer.py - ভিজ্যুয়ালাইজেশন ফাংশন
"""
import cv2
import numpy as np
from foatball.config import *
from foatball.utilitis.utils import *

class FootballVisualizer:
    """ফুটবল বিশ্লেষণ ভিজ্যুয়ালাইজেশন"""
    
    def __init__(self, tracker):
        self.tracker = tracker
    
    def draw_ball_owner(self, frame, owner_id, owner_position, owner_team):
        """
        বল মালিক ভিজ্যুয়ালাইজ করে
        """
        if owner_id is None or owner_position is None:
            return frame
        
        x, y = int(owner_position[0]), int(owner_position[1])
        
        # মালিকের চারপাশে বৃত্ত
        color = COLORS['TEAM_A'] if owner_team == "A" else COLORS['TEAM_B']
        cv2.circle(frame, (x, y), 35, color, 3)
        
        # টেক্সট
        text = f"Owner: P{owner_id}"
        frame = draw_text_with_background(
            frame, text, (x - 40, y - 40),
            font_scale=0.6, text_color=(255, 255, 255), bg_color=color
        )
        
        return frame
    
    def draw_pass_arrow(self, frame, pass_event):
        """
        পাসের তীর আঁকে
        """
        if pass_event is None:
            return frame
        
        from_pos = pass_event['from_position']
        to_pos = pass_event['to_position']
        
        if from_pos is None or to_pos is None:
            return frame
        
        # কালার নির্ধারণ
        if pass_event['successful']:
            color = COLORS['SUCCESSFUL_PASS']
            thickness = 3
        else:
            color = COLORS['FAILED_PASS']
            thickness = 2
        
        # তীর আঁকা
        cv2.arrowedLine(
            frame,
            tuple(map(int, from_pos)),
            tuple(map(int, to_pos)),
            color,
            thickness,
            tipLength=0.2,
            line_type=cv2.LINE_AA
        )
        
        # পাসের তথ্য
        mid_x = int((from_pos[0] + to_pos[0]) / 2)
        mid_y = int((from_pos[1] + to_pos[1]) / 2)
        
        info_text = f"{pass_event['time']:.2f}s"
        frame = draw_text_with_background(
            frame, info_text, (mid_x - 20, mid_y - 15),
            font_scale=0.5, text_color=(255, 255, 255), bg_color=color
        )
        
        return frame
    
    def draw_stats_panel(self, frame, stats):
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
    
    def draw_pass_history(self, frame, pass_history):
        """
        পাস ইতিহাস আঁকে
        """
        height, width = frame.shape[:2]
        
        # প্যানেল
        panel_x = width - 270
        cv2.rectangle(frame, (panel_x, 12), (width - 12, 200), (0, 0, 0), -1)
        
        # টাইটেল
        cv2.putText(
            frame, "RECENT PASSES", (panel_x + 15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 155), 1
        )
        
        # প্রতিটি পাস
        y_offset = 60
        recent_passes = pass_history[-5:] if pass_history else []
        
        for i, pass_event in enumerate(recent_passes):
            from_player = pass_event.get('from_player', '?')
            to_player = pass_event.get('to_player', '?')
            duration = pass_event.get('time', 0)
            successful = pass_event.get('successful', False)
            
            # কালার নির্ধারণ
            if successful:
                color = (0, 255, 0)  # সবুজ
            else:
                color = (255, 0, 0)  # লাল
            
            text = f"P{from_player}→P{to_player} ({duration:.2f}s)"
            
            cv2.putText(
                frame, text, (panel_x + 10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        
        return frame
    
    def draw_frame_info(self, frame, frame_idx, fps):
        """
        ফ্রেম তথ্য আঁকে
        """
        current_time = frame_idx / fps
        
        # সময় এবং ফ্রেম নম্বর
        time_text = f"Time:{current_time:.1f}s | Frame:{frame_idx}"
        cv2.putText(
            frame, time_text, (frame.shape[1] - 480, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 255, 250), 2
        )
        
        return frame