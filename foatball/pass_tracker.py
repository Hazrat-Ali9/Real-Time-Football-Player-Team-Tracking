"""
pass_tracker.py - পাস এবং মালিকানা ট্র্যাকিং লজিক
"""
import numpy as np
from collections import defaultdict, deque
from foatball.config import *
from foatball.utilitis.utils import *

class PassTracker:
    """পাস এবং বল মালিকানা ট্র্যাক করে"""
    
    def __init__(self, fps=30):
        self.fps = fps
        
        # বল মালিকানা ট্র্যাকিং
        self.current_owner = None
        self.current_owner_position = None
        self.current_owner_team = None
        self.ownership_start_time = 0
        self.ownership_start_frame = 0
        
        # পাস ইভেন্ট ট্র্যাকিং
        self.last_owner = None
        self.last_owner_position = None
        self.last_owner_team = None
        self.last_change_time = 0
        
        # পরিসংখ্যান
        self.pass_events = []  # সব পাস
        self.recent_passes = deque(maxlen=MAX_PASS_HISTORY)  # সাম্প্রতিক পাস
        
        self.possession_stats = {
            'team_a': {'frames': 0, 'time': 0.0},
            'team_b': {'frames': 0, 'time': 0.0},
            'no_possession': {'frames': 0, 'time': 0.0}
        }
        
        self.pass_stats = {
            'total': 0,
            'successful': 0,
            'team_a_internal': 0,
            'team_b_internal': 0,
            'interceptions': 0,
            'lost_balls': 0
        }
        
        # খেলোয়াড়ভিত্তিক পরিসংখ্যান
        self.player_stats = defaultdict(lambda: {
            'possession_frames': 0,
            'passes_made': 0,
            'passes_received': 0,
            'successful_passes': 0
        })
    
    def find_ball_owner(self, ball_detections, player_detections, frame_idx):
        """
        বলের বর্তমান মালিক খুঁজে বের করে
        """
        if len(ball_detections) == 0 or len(player_detections) == 0:
            return None, None, None
        
        # সবচেয়ে কনফিডেন্ট বল ডিটেকশন নিন
        if len(ball_detections) > 0:
            ball_box = ball_detections.xyxy[0]
            ball_center = box_center(ball_box)
        else:
            return None, None, None
        
        # সবচেয়ে কাছের খেলোয়াড় খুঁজুন
        min_distance = float('inf')
        owner_id = None
        owner_position = None
        owner_team = None
        
        for i, (box, tracker_id, class_id) in enumerate(zip(
            player_detections.xyxy, 
            player_detections.tracker_id, 
            player_detections.class_id
        )):
            if tracker_id is None:
                continue
            
            player_center = box_center(box)
            distance = calculate_distance(ball_center, player_center)
            
            if distance < min_distance and distance < BALL_DISTANCE_THRESHOLD:
                min_distance = distance
                owner_id = tracker_id
                owner_position = player_center
                owner_team = get_player_team(class_id)
        
        return owner_id, owner_position, owner_team
    
    def update_possession(self, owner_id, owner_position, owner_team, frame_idx, current_time):
        """
        মালিকানা আপডেট করে এবং পাস চেক করে
        """
        # প্রথম ফ্রেম বা মালিক নেই
        if self.current_owner is None and owner_id is None:
            self.current_owner = owner_id
            self.current_owner_position = owner_position
            self.current_owner_team = owner_team
            self.ownership_start_time = current_time
            self.ownership_start_frame = frame_idx
            return None
        
        # মালিক বদলালো
        if owner_id != self.current_owner:
            # পূর্ববর্তী মালিকের পরিসংখ্যান আপডেট
            if self.current_owner is not None:
                possession_duration = current_time - self.ownership_start_time
                self._update_player_possession_stats(
                    self.current_owner, possession_duration
                )
            
            # পাস ইভেন্ট চেক
            pass_event = self._check_pass_event(
                owner_id, owner_position, owner_team, 
                frame_idx, current_time
            )
            
            # নতুন মালিক সেট করুন
            self.current_owner = owner_id
            self.current_owner_position = owner_position
            self.current_owner_team = owner_team
            self.ownership_start_time = current_time
            self.ownership_start_frame = frame_idx
            
            return pass_event
        
        # মালিক একই আছে
        return None
    
    def _check_pass_event(self, new_owner, new_position, new_team, frame_idx, current_time):
        """
        পাস ইভেন্ট চেক করে
        """
        if self.last_owner is None or new_owner is None:
            # প্রথম বার বা বল কারো কাছে নেই
            self.last_owner = new_owner
            self.last_owner_position = new_position
            self.last_owner_team = new_team
            self.last_change_time = current_time
            return None
        
        # সময়ের পার্থক্য
        time_diff = current_time - self.last_change_time
        
        # পাস হয়েছে কিনা চেক
        if time_diff < PASS_TIME_THRESHOLD and new_owner != self.last_owner:
            # পাসের দূরত্ব
            pass_distance = calculate_distance(
                self.last_owner_position, new_position
            ) if self.last_owner_position is not None and new_position is not None else 0
            
            # পাসের ধরন
            pass_type = self._determine_pass_type(
                self.last_owner_team, new_team, time_diff, pass_distance
            )
            
            # পাস ইভেন্ট তৈরি
            pass_event = {
                'from_player': self.last_owner,
                'to_player': new_owner,
                'from_team': self.last_owner_team,
                'to_team': new_team,
                'from_position': self.last_owner_position,
                'to_position': new_position,
                'time': time_diff,
                'distance': pass_distance,
                'type': pass_type,
                'frame': frame_idx,
                'timestamp': current_time,
                'successful': self._is_pass_successful(time_diff, pass_distance)
            }
            
            # পরিসংখ্যান আপডেট
            self._update_pass_stats(pass_event)
            
            # খেলোয়াড় পরিসংখ্যান আপডেট
            self._update_player_pass_stats(pass_event)
            
            # ইতিহাসে যোগ
            self.pass_events.append(pass_event)
            self.recent_passes.append(pass_event)
            
            # আউটপুট প্রিন্ট
            self._print_pass_info(pass_event)
            
            # আপডেট
            self.last_owner = new_owner
            self.last_owner_position = new_position
            self.last_owner_team = new_team
            self.last_change_time = current_time
            
            return pass_event
        
        # আপডেট (পাস না হলেও)
        self.last_owner = new_owner
        self.last_owner_position = new_position
        self.last_owner_team = new_team
        self.last_change_time = current_time
        
        return None
    
    def _determine_pass_type(self, from_team, to_team, time_diff, distance):
        """পাসের ধরন নির্ধারণ"""
        # দলভিত্তিক
        if from_team == to_team:
            if from_team == "A":
                base_type = "TEAM_A_PASS"
            else:
                base_type = "TEAM_B_PASS"
        else:
            if from_team == "A" and to_team == "B":
                return "INTERCEPTION_BY_B"
            elif from_team == "B" and to_team == "A":
                return "INTERCEPTION_BY_A"
            else:
                base_type = "UNKNOWN_PASS"
        
        # সময়ভিত্তিক
        if time_diff < 0.3:
            time_type = "QUICK"
        elif time_diff < 0.6:
            time_type = "NORMAL"
        else:
            time_type = "SLOW"
        
        # দূরত্বভিত্তিক
        if distance > 200:
            dist_type = "LONG"
        else:
            dist_type = "SHORT"
        
        return f"{time_type}_{dist_type}_{base_type}"
    
    def _is_pass_successful(self, time_diff, distance):
        """পাস সফল কিনা চেক"""
        # সহজ লজিক: দ্রুত এবং মাঝারি দূরত্বের পাস সফল
        return time_diff < 0.5 and 50 < distance < 300
    
    def _update_pass_stats(self, pass_event):
        """পাস পরিসংখ্যান আপডেট"""
        self.pass_stats['total'] += 1
        
        if pass_event['successful']:
            self.pass_stats['successful'] += 1
        
        if pass_event['from_team'] == pass_event['to_team']:
            if pass_event['from_team'] == "A":
                self.pass_stats['team_a_internal'] += 1
            else:
                self.pass_stats['team_b_internal'] += 1
        else:
            self.pass_stats['interceptions'] += 1
    
    def _update_player_possession_stats(self, player_id, duration):
        """খেলোয়াড়ের মালিকানা পরিসংখ্যান আপডেট"""
        if player_id is not None:
            self.player_stats[player_id]['possession_frames'] += duration * self.fps
    
    def _update_player_pass_stats(self, pass_event):
        """খেলোয়াড়ের পাস পরিসংখ্যান আপডেট"""
        from_player = pass_event['from_player']
        to_player = pass_event['to_player']
        
        if from_player is not None:
            self.player_stats[from_player]['passes_made'] += 1
            if pass_event['successful']:
                self.player_stats[from_player]['successful_passes'] += 1
        
        if to_player is not None:
            self.player_stats[to_player]['passes_received'] += 1
    
    def _print_pass_info(self, pass_event):
        """পাস তথ্য প্রিন্ট"""
        icons = {
            "TEAM_A_PASS": "🔴",
            "TEAM_B_PASS": "🔵",
            "INTERCEPTION": "🟡",
            "QUICK": "⚡",
            "NORMAL": "➡️",
            "SLOW": "🐢"
        }
        
        # আইকন নির্বাচন
        icon = "⚽"
        for key in icons:
            if key in pass_event['type']:
                icon = icons[key]
                break
        
        print(f"{icon} PASS: Player {pass_event['from_player']} → Player {pass_event['to_player']}")
        print(f"   Type: {pass_event['type']}")
        print(f"   Time: {pass_event['time']:.3f}s | Distance: {pass_event['distance']:.1f}px")
        print(f"   Successful: {'✅' if pass_event['successful'] else '❌'}")
        print("-" * 40)
    
    def get_current_stats(self):
        """বর্তমান পরিসংখ্যান রিটার্ন করে"""
        total_frames = sum(team['frames'] for team in self.possession_stats.values())
        
        if total_frames == 0:
            team_a_percent = team_b_percent = 0
        else:
            team_a_percent = (self.possession_stats['team_a']['frames'] / total_frames) * 100
            team_b_percent = (self.possession_stats['team_b']['frames'] / total_frames) * 100
        
        # গড় পাস সময়
        avg_pass_time = 0
        if self.pass_stats['total'] > 0 and self.pass_events:
            avg_pass_time = sum(p['time'] for p in self.pass_events) / len(self.pass_events)
        
        # সফলতার হার
        success_rate = 0
        if self.pass_stats['total'] > 0:
            success_rate = (self.pass_stats['successful'] / self.pass_stats['total']) * 100
        
        return {
            'team_a_possession': f"{team_a_percent:.1f}%",
            'team_b_possession': f"{team_b_percent:.1f}%",
            'total_passes': self.pass_stats['total'],
            'successful_passes': self.pass_stats['successful'],
            'success_rate': f"{success_rate:.1f}%",
            'team_a_passes': self.pass_stats['team_a_internal'],
            'team_b_passes': self.pass_stats['team_b_internal'],
            'interceptions': self.pass_stats['interceptions'],
            'avg_pass_time': f"{avg_pass_time:.3f}s",
            'current_owner': self.current_owner,
            'current_owner_team': self.current_owner_team,
            'recent_passes': list(self.recent_passes)[-5:]  # শেষ ৫টি পাস
        }
    
    def print_summary(self):
        """সারাংশ প্রিন্ট"""
        stats = self.get_current_stats()
        
        print("\n" + "="*60)
        print("PASS ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📊 Possession:")
        print(f"  Team A: {stats['team_a_possession']}")
        print(f"  Team B: {stats['team_b_possession']}")
        
        print(f"\n🎯 Passing Statistics:")
        print(f"  Total Passes: {stats['total_passes']}")
        print(f"  Successful: {stats['successful_passes']}")
        print(f"  Success Rate: {stats['success_rate']}")
        print(f"  Team A Passes: {stats['team_a_passes']}")
        print(f"  Team B Passes: {stats['team_b_passes']}")
        print(f"  Interceptions: {stats['interceptions']}")
        print(f"  Avg Pass Time: {stats['avg_pass_time']}")
        
        print(f"\n👤 Player Statistics (Top 5):")
        sorted_players = sorted(
            self.player_stats.items(), 
            key=lambda x: x[1]['passes_made'], 
            reverse=True
        )[:5]
        
        for player_id, p_stats in sorted_players:
            if p_stats['passes_made'] > 0:
                success_rate = (p_stats['successful_passes'] / p_stats['passes_made']) * 100
                print(f"  Player {player_id}: {p_stats['passes_made']} passes, "
                      f"{p_stats['successful_passes']} successful ({success_rate:.1f}%)")
        
        print("="*60)