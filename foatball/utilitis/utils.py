"""
utils.py - সাধারণ সাহায্যকারী ফাংশন
"""
import numpy as np
import cv2

def box_center(xyxy):
    """
    বাউন্ডিং বক্সের কেন্দ্র বের করে
    """
    x1, y1, x2, y2 = xyxy
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def calculate_distance(point1, point2):
    """
    দুটি পয়েন্টের মধ্যে ইউক্লিডীয় দূরত্ব বের করে
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))

def get_player_team(player_class_id):
    """
    খেলোয়াড়ের দল নির্ধারণ করে
    """
    return "A" if player_class_id == 0 else "B"

def calculate_pass_type(duration, distance):
    """
    পাসের ধরন নির্ধারণ করে
    """
    if duration < 0.3:
        if distance > 200:
            return "QUICK_LONG_PASS"
        else:
            return "QUICK_SHORT_PASS"
    elif duration < 0.6:
        if distance > 200:
            return "NORMAL_LONG_PASS"
        else:
            return "NORMAL_SHORT_PASS"
    else:
        return "SLOW_PASS"

def draw_text_with_background(frame, text, position, font_scale=0.6, 
                             text_color=(255, 255, 255), bg_color=(0, 0, 0),
                             thickness=1):
    """
    ব্যাকগ্রাউন্ড সহ টেক্সট আঁকে
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # ব্যাকগ্রাউন্ড রেকটেঙ্গেল
    bg_x1 = position[0] - 5
    bg_y1 = position[1] - text_size[1] - 5
    bg_x2 = position[0] + text_size[0] + 5
    bg_y2 = position[1] + 5
    
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
    
    # টেক্সট
    cv2.putText(frame, text, position, font, font_scale, text_color, thickness)
    
    return frame