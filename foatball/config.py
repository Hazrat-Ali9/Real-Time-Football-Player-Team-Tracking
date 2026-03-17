"""
config.py - সব সেটিংস এবং থ্রেশহোল্ড
"""

# থ্রেশহোল্ড সেটিংস
PASS_TIME_THRESHOLD = 1.0      # পাস হিসেবে গণ্য করার সর্বোচ্চ সময় (সেকেন্ড)
BALL_DISTANCE_THRESHOLD = 80   # বলের মালিক হবার সর্বোচ্চ দূরত্ব (পিক্সেল)
MIN_PASS_DISTANCE = 50         # পাসের ন্যূনতম দূরত্ব (পিক্সেল)
SUCCESSFUL_PASS_HOLD_TIME = 0.5  # সফল পাসের ন্যূনতম ধরে রাখার সময়

# কালার কোড (BGR ফরম্যাট)
COLORS = {
    'TEAM_A': (0, 0, 255),        # লাল
    'TEAM_B': (255, 0, 0),        # নীল
    'BALL': (0, 255, 255),        # হলুদ
    'QUICK_PASS': (0, 255, 0),    # সবুজ (দ্রুত পাস)
    'NORMAL_PASS': (255, 255, 0), # হলুদ (সাধারণ পাস)
    'SUCCESSFUL_PASS': (0, 255, 0), # সবুজ (সফল পাস)
    'FAILED_PASS': (0, 0, 255),   # লাল (ব্যর্থ পাস)
    'INTERCEPTION': (0, 165, 255) # কমলা (ইন্টারসেপশন)
}

# ভিজ্যুয়ালাইজেশন সেটিংস
SHOW_PASS_ARROWS = True
SHOW_POSSESSION_STATS = True
SHOW_PASS_HISTORY = True
MAX_PASS_HISTORY = 10

# দল সেটিংস (আপনার মডেল অনুযায়ী)
CLASS_IDS = {
    'BALL': 0,
    'GOALKEEPER': 1,
    'PLAYER': 2,
    'REFEREE': 3
}

TEAM_NAMES = {
    0: "Team Blue",
    1: "Team Pink"
}