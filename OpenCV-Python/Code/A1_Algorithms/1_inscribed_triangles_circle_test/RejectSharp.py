import numpy as np


class SharpTurn:
    def __init__(self):
        self.new_edgeList = []
        self.new_segList = []


"""
    拒绝锐角算法
"""


def reject_sharp_turn(edgeList, segList, angle):
    result = SharpTurn()
    no_seg_grps = len(segList)
    break_points = []
    Threshold_theta = np.pi / 2
    new_segList1, new_edgeList1 = [], []

    for ii in range(no_seg_grps):
        present_seg_grp = segList[ii]
        no_of_seg = len(present_seg_grp) - 1

        present_vector = (
            present_seg_grp[1][0] - present_seg_grp[0][0],
            present_seg_grp[1][1] - present_seg_grp[0][1],
        )

        for seg_no in range(no_of_seg - 1):
            length_present_vector = np.sqrt(
                present_vector[0] ** 2 + present_vector[1] ** 2
            )
            next_vector = (
                present_seg_grp[seg_no + 2][0] - present_seg_grp[seg_no + 1][0],
                present_seg_grp[seg_no + 2][1] - present_seg_grp[seg_no + 1][1],
            )
            length_next_vector = np.sqrt(next_vector[0] ** 2 + next_vector[1] ** 2)
            cos_pre_next = (
                present_vector[0] * next_vector[0] + present_vector[1] * next_vector[1]
            ) / (length_present_vector * length_next_vector)

            if cos_pre_next <= np.cos(angle / 180.0 * np.pi):
                break_points.append((ii, seg_no + 1))

            present_vector = next_vector

    if not break_points:
        result.new_edgeList = edgeList
        result.new_segList = segList
        return result

    index = 0
    current_break = break_points[index][0]

    for ii in range(no_seg_grps):
        current_seg = segList[ii]
        current_edge = edgeList[ii]

        if len(current_seg) > 2:
            if ii == current_break:
                count = 1
                first_edge_index, last_edge_index, first_seg_index, last_seg_index = (
                    0,
                    0,
                    0,
                    0,
                )

                while ii == current_break:
                    if count == 1:
                        first_seg_index = 0
                        first_edge_index = 0
                    else:
                        first_seg_index = last_seg_index
                        first_edge_index = last_edge_index

                    last_seg_index = break_points[index][1]

                    for jj in range(first_edge_index, len(current_edge)):
                        if current_seg[last_seg_index][0] == current_edge[jj][0] and current_seg[last_seg_index][1] == current_edge[jj][1]:
                            last_edge_index = jj
                            break

                    if last_seg_index - first_seg_index >= 1:
                        block_seg = current_seg[first_seg_index : last_seg_index + 1]
                        block_edge = current_edge[
                            first_edge_index : last_edge_index + 1
                        ]
                        new_edgeList1.append(block_edge)
                        new_segList1.append(block_seg)

                    index += 1
                    if index > len(break_points) - 1:
                        break

                    current_break = break_points[index][0]
                    count += 1

                if len(current_seg) - last_seg_index >= 2:
                    block1_seg = current_seg[last_seg_index:]
                    block1_edge = current_edge[last_edge_index:]
                    new_edgeList1.append(block1_edge)
                    new_segList1.append(block1_seg)
            else:
                new_edgeList1.append(edgeList[ii])
                new_segList1.append(segList[ii])

    result.new_edgeList = new_edgeList1
    result.new_segList = new_segList1
    return result
