
## Final implement

"""
utils.py - সাধারণ সাহায্যকারী ফাংশন
"""
import numpy as np
import cv2

def calculate_distance(point1, point2):
    """
    দুটি পয়েন্টের মধ্যে ইউক্লিডীয় দূরত্ব বের করে
    
    Args:
        point1: প্রথম পয়েন্ট [x, y]
        point2: দ্বিতীয় পয়েন্ট [x, y]
    
    Returns:
        float: দূরত্ব (পিক্সেল)
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def calculate_pass_type(duration, distance):
    """
    পাসের ধরন নির্ধারণ করে
    
    Args:
        duration: পাসের সময় (সেকেন্ড)
        distance: পাসের দূরত্ব (পিক্সেল)
    
    Returns:
        str: পাসের ধরন
    """
    from config import QUICK_PASS_THRESHOLD, DISTANCE_THRESHOLD
    
    if duration < QUICK_PASS_THRESHOLD:
        if distance is not None and distance > DISTANCE_THRESHOLD:
            return "QUICK_LONG_PASS"
        else:
            return "QUICK_SHORT_PASS"
    else:
        if distance is not None and distance > DISTANCE_THRESHOLD:
            return "NORMAL_LONG_PASS"
        else:
            return "NORMAL_SHORT_PASS"


def print_pass_info(pass_info):
    """
    পাসের তথ্য সুন্দরভাবে প্রিন্ট করে
    """
    colors = {
        "QUICK_LONG_PASS": "🟢",
        "QUICK_SHORT_PASS": "🟡",
        "NORMAL_LONG_PASS": "🔵",
        "NORMAL_SHORT_PASS": "⚪"
    }
    
    color = colors.get(pass_info['pass_type'], "⚪")
    distance_str = f"{pass_info['distance']:.1f}px" if pass_info['distance'] else "N/A"
    
    print(f"{color} PASS: Player {pass_info['from_player']} → Player {pass_info['to_player']}")
    print(f"   Type: {pass_info['pass_type']}")
    print(f"   Time: {pass_info['time']:.3f}s | Distance: {distance_str}")
    print(f"   Frame Time: {pass_info['frame_time']:.2f}s")
    print("-" * 40)





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