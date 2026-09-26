"""
test_label_parser.py
Kiểm tra giải thuật phân tích nhãn OBB tự động.
"""
import os
import math
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

LABEL_MAP = {
    0: 'airplane', 1: 'baseball_diamond', 2: 'bridge', 3: 'ground_track_field',
    4: 'small_vehicle', 5: 'large_vehicle', 6: 'ship', 7: 'tennis_court',
    8: 'basketball_court', 9: 'storage_tank', 10: 'soccer_ball_field',
    11: 'roundabout', 12: 'harbor', 13: 'swimming_pool', 14: 'helicopter'
}

def parse_dota_label(filepath, img_w=1024, img_h=1024):
    objects = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            parts = [float(x) for x in line.strip().split()]
            if len(parts) < 9:
                continue
            cid = int(parts[0])
            cname = LABEL_MAP.get(cid, f'class_{cid}')
            pts = [(parts[k]*img_w, parts[k+1]*img_h) for k in range(1, 9, 2)]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            cx, cy = sum(xs)/4.0, sum(ys)/4.0
            
            # Ô lưới 3x3
            col = 'A' if cx < img_w/3.0 else ('B' if cx < 2.0*img_w/3.0 else 'C')
            row = '1' if cy < img_h/3.0 else ('2' if cy < 2.0*img_h/3.0 else '3')
            cell = f'{col}{row}'
            
            bbox = [
                round(max(0.0, min(xs)), 1),
                round(max(0.0, min(ys)), 1),
                round(min(float(img_w), max(xs)), 1),
                round(min(float(img_h), max(ys)), 1)
            ]
            
            # Tính hướng trục chính từ 4 đỉnh OBB
            # Cạnh 1: p0 -> p1
            dx1, dy1 = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
            len1 = math.hypot(dx1, dy1)
            # Cạnh 2: p1 -> p2
            dx2, dy2 = pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]
            len2 = math.hypot(dx2, dy2)
            
            if len1 >= len2:
                angle_rad = math.atan2(dy1, dx1)
                major_len = len1
                minor_len = len2
            else:
                angle_rad = math.atan2(dy2, dx2)
                major_len = len2
                minor_len = len1
                
            angle_deg = math.degrees(angle_rad) % 180.0
            # 2 bin: 0..90 độ là Đông Bắc - Tây Nam, 90..180 độ là Tây Bắc - Đông Nam
            direction = 'Đông Bắc - Tây Nam' if (angle_deg < 90.0) else 'Tây Bắc - Đông Nam'
            
            objects.append({
                'obj_id': line_idx,
                'class_id': cname,
                'pts': pts,
                'bbox': bbox,
                'center': (round(cx, 1), round(cy, 1)),
                'cell': cell,
                'major_len_px': round(major_len, 1),
                'minor_len_px': round(minor_len, 1),
                'angle_deg': round(angle_deg, 1),
                'direction': direction
            })
    return objects

if __name__ == '__main__':
    label_dir = r'C:\Users\trang\Desktop\rgsp_nckh\rs-solve\labels'
    for f in ['dota_P1053.txt', 'dota_P1142.txt', 'dota_P1470.txt']:
        p = os.path.join(label_dir, f)
        objs = parse_dota_label(p)
        print(f'=== {f} ({len(objs)} objects) ===')
        for o in objs[:4]:
            print(f"  [{o['class_id']}] cell={o['cell']}, bbox={o['bbox']}, dir={o['direction']}, angle={o['angle_deg']} deg")
