import numpy as np

"""
    计算点到直线的距离
"""


def perpendicular_distance(pt, line_start, line_end):
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    # 归一化
    mag = np.sqrt(dx**2 + dy**2)
    if mag > 0.0:
        dx /= mag
        dy /= mag
    pvx = pt[0] - line_start[0]
    pvy = pt[1] - line_start[1]
    pvdot = dx * pvx + dy * pvy
    dsx = pvdot * dx
    dsy = pvdot * dy
    ax = pvx - dsx
    ay = pvy - dsy
    return np.sqrt(ax**2 + ay**2)

"""
    RDP算法
"""


def ramer_douglas_peucker(point_list, epsilon, out):
    if len(point_list) < 2:
        raise ValueError("Not enough points to simplify")
    dmax = 0.0
    index = 0
    end = len(point_list) - 1
    for i in range(1, end):
        d = perpendicular_distance(point_list[i], point_list[0], point_list[end])
        if d > dmax:
            index = i
            dmax = d
    if dmax > epsilon:
        rec_results1 = []
        rec_results2 = []
        first_line = point_list[: index + 1]
        last_line = point_list[index:]
        ramer_douglas_peucker(first_line, epsilon, rec_results1)
        ramer_douglas_peucker(last_line, epsilon, rec_results2)
        out.extend(rec_results1[:-1])
        out.extend(rec_results2)
        if len(out) < 2:
            raise RuntimeError("Problem assembling output")
    else:
        out.clear()
        out.append(point_list[0])
        out.append(point_list[end])