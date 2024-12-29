import math

def calculate_direction(traj):
    if len(traj) < 2:
        return "Unknown"

    # Tính vector quỹ đạo giữa hai điểm cuối
    x1, y1 = traj[-2]
    x2, y2 = traj[-1]
    dx, dy = x2 - x1, y2 - y1

    # Tính góc
    angle = math.degrees(math.atan2(dy, dx))

    # Dự đoán hướng dựa trên góc
    if -45 <= angle <= 45:
        return "Right"
    elif 45 < angle <= 135:
        return "Up"
    elif -135 <= angle < -45:
        return "Left"
    else:
        return "Down"