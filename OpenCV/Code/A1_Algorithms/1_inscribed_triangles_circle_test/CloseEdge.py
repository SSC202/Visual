import numpy as np

class ClosedEdgesExtract:
    def __init__(self):
        self.closed_edges = []
        self.not_closed_edges = []

"""
    检查边缘闭合性质
"""
def extract_closed_edges(edge_list):
    edges = ClosedEdgesExtract()
    for edge in edge_list:
        st = edge[0]  # 起点
        ed = edge[-1]  # 终点

        # 计算起点和终点之间的距离
        dist = np.sqrt((st[0] - ed[0]) ** 2 + (st[1] - ed[1]) ** 2)

        # 如果距离小于等于3，则将该边缘视为闭合边缘
        if dist <= 3:
            edges.closed_edges.append(edge)  # 将闭合边缘添加到闭合边缘列表中
        else:  # 否则将该边缘视为非闭合边缘
            edges.not_closed_edges.append(edge)  # 将非闭合边缘添加到非闭合边缘列表中
    return edges