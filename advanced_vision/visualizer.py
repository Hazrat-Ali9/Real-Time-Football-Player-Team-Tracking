"""
visualizer.py - ভিজ্যুয়ালাইজেশন ফাংশন
"""
import cv2
import numpy as np
from config import COLORS


class FootballVisualizer:
    """ফুটবল বিশ্লেষণ ভিজ্যুয়ালাইজেশন"""
    
    def __init__(self):
        self.pass_history = []
    
    def draw_player(self, frame, player_id, position, team, has_ball=False):
        """
        খেলোয়াড় আঁকে
        
        Args:
            frame: ভিডিও ফ্রেম
            player_id: খেলোয়াড় আইডি
            position: অবস্থান (x, y)
            team: দল ('A' বা 'B')
            has_ball: বল আছে কিনা
        
        Returns:
            numpy array: আপডেটেড ফ্রেম
        """
        if position is None:
            return frame
        
        x, y = int(position[0]), int(position[1])
        
        # রং নির্ধারণ
        if team == 'A':
            color = COLORS['TEAM_A']  # লাল
        else:
            color = COLORS['TEAM_B']  # নীল
        
        # বৃত্ত আঁকা
        radius = 25 if has_ball else 20
        thickness = 3 if has_ball else 2
        
        cv2.circle(frame, (x, y), radius, color, thickness)
        
        # খেলোয়াড় নম্বর
        cv2.putText(
            frame, str(player_id), 
            (x - 10, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        # বল থাকলে ছোট বৃত্ত
        if has_ball:
            cv2.circle(frame, (x, y - radius - 5), 5, (0, 255, 255), -1)
        
        return frame
    
    def draw_pass_arrow(self, frame, from_pos, to_pos, pass_type, duration):
        """
        পাসের তীর আঁকে
        
        Args:
            frame: ভিডিও ফ্রেম
            from_pos: পাস শুরু অবস্থান
            to_pos: পাস শেষ অবস্থান
            pass_type: পাসের ধরন
            duration: পাসের সময়
        
        Returns:
            numpy array: আপডেটেড ফ্রেম
        """
        if from_pos is None or to_pos is None:
            return frame
        
        # কালার নির্ধারণ
        if duration < 0.5:
            color = COLORS['QUICK_PASS']
            thickness = 3
        else:
            color = COLORS['NORMAL_PASS']
            thickness = 2
        
        # তীর আঁকা
        cv2.arrowedLine(
            frame,
            tuple(map(int, from_pos)),
            tuple(map(int, to_pos)),
            color,
            thickness,
            tipLength=0.2
        )
        
        # পাসের সময় দেখানো
        mid_x = int((from_pos[0] + to_pos[0]) / 2)
        mid_y = int((from_pos[1] + to_pos[1]) / 2)
        
        cv2.putText(
            frame, f"{duration:.2f}s",
            (mid_x - 20, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
        
        return frame
    
    def draw_stats_panel(self, frame, stats):
        """
        পরিসংখ্যান প্যানেল আঁকে
        
        Args:
            frame: ভিডিও ফ্রেম
            stats: পরিসংখ্যান ডিকশনারি
        
        Returns:
            numpy array: আপডেটেড ফ্রেম
        """
        height, width = frame.shape[:2]
        
        # আধা-স্বচ্ছ প্যানেল তৈরি
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # টেক্সট লিস্ট
        stats_text = [
            "⚽ LIVE STATS",
            f"Team A: {stats.get('team_a_percent', 0):.1f}%",
            f"Team B: {stats.get('team_b_percent', 0):.1f}%",
            f"Passes: {stats.get('total_passes', 0)}",
            f"Successful: {stats.get('successful_passes', 0)}",
            f"Avg Pass: {stats.get('avg_pass_time', 0):.2f}s"
        ]
        
        # টেক্সট আঁকা
        y_offset = 40
        line_height = 25
        
        for i, text in enumerate(stats_text):
            color = (0, 255, 155) if i == 0 else (200, 200, 200)
            font_size = 0.6 if i == 0 else 0.5
            
            cv2.putText(
                frame, text,
                (20, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 1
            )
        
        return frame
    
    def draw_pass_history(self, frame, pass_history):
        """
        পাসের ইতিহাস আঁকে
        
        Args:
            frame: ভিডিও ফ্রেম
            pass_history: পাস ইভেন্টের তালিকা
        
        Returns:
            numpy array: আপডেটেড ফ্রেম
        """
        height, width = frame.shape[:2]
        
        # প্যানেল আঁকা
        panel_x = width - 250
        cv2.rectangle(frame, (panel_x, 10), (width - 10, 180), (0, 0, 0), -1)
        
        # টাইটেল
        cv2.putText(
            frame, "RECENT PASSES",
            (panel_x + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        
        # প্রতিটি পাস দেখানো
        y_offset = 60
        recent_passes = pass_history[-5:]  # সর্বশেষ ৫টি পাস
        
        for i, pass_event in enumerate(recent_passes):
            from_player = pass_event.get('from_player', '?')
            to_player = pass_event.get('to_player', '?')
            duration = pass_event.get('time', 0)
            
            # কালার নির্ধারণ
            if duration < 0.5:
                color = (0, 255, 0)  # সবুজ
            elif duration < 1.0:
                color = (255, 255, 0)  # হলুদ
            else:
                color = (255, 0, 0)  # লাল
            
            text = f"{from_player} → {to_player} ({duration:.2f}s)"
            
            cv2.putText(
                frame, text,
                (panel_x + 10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
        
        return frame