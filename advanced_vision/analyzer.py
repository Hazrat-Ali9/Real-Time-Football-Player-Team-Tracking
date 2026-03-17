"""
analyzer.py - পাস এবং মালিকানা বিশ্লেষণ করে
"""
from config import PASS_TIME_THRESHOLD, TEAM_A_PLAYERS, TEAM_B_PLAYERS
from utils import calculate_distance, calculate_pass_type, print_pass_info


class PassAnalyzer:
    """পাস বিশ্লেষণ করে"""
    
    def __init__(self, fps=30):
        self.fps = fps
        
        # ট্র্যাকিং ভেরিয়েবল
        self.last_owner = None
        self.last_owner_position = None
        self.last_change_time = None
        
        # পরিসংখ্যান
        self.pass_events = []
        self.possession_stats = {
            'team_a': 0,
            'team_b': 0,
            'no_owner': 0
        }
    
    def detect_pass(self, owner_id, owner_position, frame_idx):
        """
        একটি ফ্রেমের জন্য পাস ডিটেক্ট করে
        
        Args:
            owner_id: বর্তমান মালিক আইডি
            owner_position: মালিকের অবস্থান
            frame_idx: ফ্রেম নম্বর
        
        Returns:
            dict: পাস ইভেন্ট বা None
        """
        current_time = frame_idx / self.fps
        
        if owner_id is None:
            if self.last_owner is not None:
                self.possession_stats['no_owner'] += 1
            return None
        
        # প্রথম বার বল মালিক পাওয়া গেছে
        if self.last_owner is None:
            self.last_owner = owner_id
            self.last_owner_position = owner_position
            self.last_change_time = current_time
            return None
        
        # যদি মালিক বদলায়
        if owner_id != self.last_owner:
            dt = current_time - self.last_change_time
            
            # পাস হয়েছে কিনা চেক
            if dt < PASS_TIME_THRESHOLD:
                # পাসের দূরত্ব ক্যালকুলেট
                pass_distance = None
                if self.last_owner_position is not None and owner_position is not None:
                    pass_distance = calculate_distance(
                        self.last_owner_position, owner_position
                    )
                
                # পাস ইভেন্ট তৈরি
                pass_event = {
                    'from_player': self.last_owner,
                    'to_player': owner_id,
                    'from_position': self.last_owner_position,
                    'to_position': owner_position,
                    'time': dt,
                    'distance': pass_distance,
                    'pass_type': calculate_pass_type(dt, pass_distance),
                    'frame_time': current_time,
                    'successful': dt < 0.5  # ডামি সফলতা চেক
                }
                
                # আউটপুট প্রিন্ট
                print_pass_info(pass_event)
                
                # পাস ইভেন্ট স্টোর
                self.pass_events.append(pass_event)
                
                # পরিসংখ্যান আপডেট
                self.update_possession_stats(owner_id)
                
                # ভেরিয়েবল আপডেট
                self.last_owner = owner_id
                self.last_owner_position = owner_position
                self.last_change_time = current_time
                
                return pass_event
        
        # মালিক বদল না হলে শুধু পজিশন আপডেট
        if owner_id == self.last_owner:
            self.last_owner_position = owner_position
        
        # পরিসংখ্যান আপডেট
        self.update_possession_stats(owner_id)
        
        return None
    
    def update_possession_stats(self, owner_id):
        """মালিকানা পরিসংখ্যান আপডেট"""
        if owner_id in TEAM_A_PLAYERS:
            self.possession_stats['team_a'] += 1
        elif owner_id in TEAM_B_PLAYERS:
            self.possession_stats['team_b'] += 1
        else:
            self.possession_stats['no_owner'] += 1
    
    def get_stats(self):
        """বর্তমান পরিসংখ্যান রিটার্ন করে"""
        total_frames = sum(self.possession_stats.values())
        
        if total_frames == 0:
            return {
                'team_a_percent': 0,
                'team_b_percent': 0,
                'total_passes': len(self.pass_events),
                'successful_passes': sum(1 for p in self.pass_events if p.get('successful')),
                'avg_pass_time': 0
            }
        
        team_a_percent = (self.possession_stats['team_a'] / total_frames) * 100
        team_b_percent = (self.possession_stats['team_b'] / total_frames) * 100
        
        avg_pass_time = 0
        if self.pass_events:
            avg_pass_time = sum(p['time'] for p in self.pass_events) / len(self.pass_events)
        
        successful_passes = sum(1 for p in self.pass_events if p.get('successful'))
        
        return {
            'team_a_percent': team_a_percent,
            'team_b_percent': team_b_percent,
            'total_passes': len(self.pass_events),
            'successful_passes': successful_passes,
            'avg_pass_time': avg_pass_time,
            'possession_team_a': self.possession_stats['team_a'],
            'possession_team_b': self.possession_stats['team_b']
        }
    
    def print_summary(self):
        """বিশ্লেষণ সারাংশ প্রিন্ট করে"""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("PASS ANALYSIS SUMMARY")
        print("="*50)
        
        print(f"\n📊 Possession:")
        print(f"  Team A: {stats['team_a_percent']:.1f}% ({stats['possession_team_a']} frames)")
        print(f"  Team B: {stats['team_b_percent']:.1f}% ({stats['possession_team_b']} frames)")
        
        print(f"\n🎯 Passing Statistics:")
        print(f"  Total Passes: {stats['total_passes']}")
        print(f"  Successful Passes: {stats['successful_passes']}")
        
        if stats['total_passes'] > 0:
            success_rate = (stats['successful_passes'] / stats['total_passes']) * 100
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Average Pass Time: {stats['avg_pass_time']:.3f}s")
        
        print("="*50)