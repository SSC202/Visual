import numpy as np


class InflexionPt:
    def __init__(self):
        self.new_edgeList = []
        self.new_segList = []


def detectInflexPt(edgeList, segList):
    result = InflexionPt()
    no_seg_grps = len(segList)

    tempSegList, tempEdgeList = [], []

    # 创建分割与索引的映射字典
    seg_index_map = {tuple(map(tuple, seg)): idx for idx, seg in enumerate(segList)}

    for present_seg_grp in segList:
        if len(present_seg_grp) <= 4:
            tempSegList.append(present_seg_grp)
            tempEdgeList.append(
                edgeList[seg_index_map[tuple(map(tuple, present_seg_grp))]]
            )
            continue

        # 计算相邻点之间的斜率角度
        theta = [
            np.arctan2(
                present_seg_grp[j + 1][0] - present_seg_grp[j][0],
                present_seg_grp[j + 1][1] - present_seg_grp[j][1],
            )
            for j in range(len(present_seg_grp) - 1)
        ]

        # 角度差异
        theta_diff = [theta[k] - theta[k + 1] for k in range(len(theta) - 1)]
        theta_diff_2pi = [diff + 2 * np.pi if diff < 0 else diff for diff in theta_diff]

        # 极性
        polarity = [1 if diff > 0 else 0 for diff in theta_diff]

        count = sum(polarity)

        if count > (len(polarity) / 2):
            polarity = [1 - p for p in polarity]

        # 检测拐点类型 "...0 1 0..."
        for i in range(len(polarity) - 2):
            if polarity[i : i + 3] == [0, 1, 0]:
                polarity[i + 1] = 0

        # 检测拐点类型 "...0 1 1 1..."
        for i in range(len(polarity) - 3):
            if polarity[i : i + 4] == [0, 1, 1, 1]:
                polarity[i + 1 : i + 4] = [0, 0, 0]

        # 记录拐点位置
        break_at_seg = [i + 2 for i, p in enumerate(polarity) if p == 0]

        if len(break_at_seg) == 0:
            tempSegList.append(present_seg_grp)
            tempEdgeList.append(
                edgeList[seg_index_map[tuple(map(tuple, present_seg_grp))]]
            )
        else:
            # 切分边缘列表和分割列表
            break_points_y_x_list = []
            for idx in break_at_seg:
                x1, y1 = present_seg_grp[idx]
                a = [
                    k
                    for k, (x, y) in enumerate(
                        edgeList[seg_index_map[tuple(map(tuple, present_seg_grp))]]
                    )
                    if x == x1 and y == y1
                ]
                if a:
                    break_points_y_x_list.append(a[0])

            seglist_temp = [present_seg_grp[: break_at_seg[0]]]
            for idx in range(1, len(break_at_seg)):
                seglist_temp.append(
                    present_seg_grp[break_at_seg[idx - 1] : break_at_seg[idx]]
                )
            seglist_temp.append(present_seg_grp[break_at_seg[-1] :])

            for seg_temp in seglist_temp:
                tempSegList.append(seg_temp)

            if break_points_y_x_list:
                edgelist_temp = [
                    edgeList[seg_index_map[tuple(map(tuple, present_seg_grp))]][
                        : break_points_y_x_list[0]
                    ]
                ]
                for idx in range(1, len(break_points_y_x_list)):
                    edgelist_temp.append(
                        edgeList[seg_index_map[tuple(map(tuple, present_seg_grp))]][
                            break_points_y_x_list[idx - 1] : break_points_y_x_list[idx]
                        ]
                    )
                edgelist_temp.append(
                    edgeList[seg_index_map[tuple(map(tuple, present_seg_grp))]][
                        break_points_y_x_list[-1] :
                    ]
                )

                for edge_temp in edgelist_temp:
                    tempEdgeList.append(edge_temp)

    result.new_edgeList = tempEdgeList
    result.new_segList = tempSegList
    return result
