import numpy as np
from scipy.linalg import pinv
from typing import List, Tuple
import cv2


def cmp(a, b):
    """
    比较函数，按线段长度从长到短排序
    """
    return len(a) > len(b)


def sort_edge_list(edgelists):
    """
    根据线段长度对线段列表进行排序
    """
    edgelists.sort(key=len, reverse=True)
    return edgelists


def com_cir_center_radius(A, B, C):
    """
    计算三个给定点构成的圆的圆心和半径
    """
    # 检查输入的有效性
    if A is None or B is None or C is None:
        return None, None

    # 计算三角形边长
    AB = np.sqrt(np.power(A[0] - B[0], 2) + np.power(A[1] - B[1], 2))
    CB = np.sqrt(np.power(C[0] - B[0], 2) + np.power(C[1] - B[1], 2))
    AC = np.sqrt(np.power(A[0] - C[0], 2) + np.power(A[1] - C[1], 2))

    # 避免出现除零错误
    if AB == 0 or CB == 0 or AC == 0:
        return None, None

    # 计算两个三角形的面积
    p = (AB + CB + AC) / 2
    S_ABC = np.sqrt(p * (p - AB) * (p - CB) * (p - AC))  # Hallen formulation

    # 避免出现除零错误
    if S_ABC == 0:
        return None, None

    # 根据面积计算半径
    R = (AB * CB * AC) / (4 * S_ABC)

    # 根据三角形顶点计算圆心
    a11 = 2 * (B[1] - A[1])
    b11 = 2 * (B[0] - A[0])
    c11 = np.power(B[1], 2) + np.power(B[0], 2) - np.power(A[1], 2) - np.power(A[0], 2)
    a12 = 2 * (C[1] - B[1])
    b12 = 2 * (C[0] - B[0])
    c12 = np.power(C[1], 2) + np.power(C[0], 2) - np.power(B[1], 2) - np.power(B[0], 2)

    # 避免出现除零错误
    if a11 * b12 - a12 * b11 == 0:
        return None, None

    O_x = (a11 * c12 - a12 * c11) / (a11 * b12 - a12 * b11)
    O_y = (c11 * b12 - c12 * b11) / (a11 * b12 - a12 * b11)

    return R, (O_x, O_y)


def two_arcs_center_radius(A1B1C1, A2B2C2, T_o, T_r):
    """
    计算两个弧的圆心和半径
    """
    flag = False
    temp1_center_radius = np.zeros(3)
    temp2_center_radius = np.zeros(3)

    # Constraint 1:
    S1 = A1B1C1[2]
    E1 = A1B1C1[-2]
    S2 = A2B2C2[2]
    E2 = A2B2C2[-2]
    M1 = A1B1C1[len(A1B1C1) // 2]
    M2 = A2B2C2[len(A2B2C2) // 2]

    K_S1E1 = (S1[0] - E1[0]) / (S1[1] - E1[1] + 1e-6)
    K_S2E2 = (S2[0] - E2[0]) / (S2[1] - E2[1] + 1e-6)

    SignM1 = np.sign(M1[0] - S1[0] - K_S1E1 * (M1[1] - S1[1]))
    SignM2 = np.sign(M2[0] - S2[0] - K_S2E2 * (M2[1] - S2[1]))
    SignS2 = np.sign(S2[0] - S1[0] - K_S1E1 * (S2[1] - S1[1]))
    SignE2 = np.sign(E2[0] - S1[0] - K_S1E1 * (E2[1] - S1[1]))
    SignS1 = np.sign(S1[0] - S2[0] - K_S2E2 * (S1[1] - S2[1]))
    SignE1 = np.sign(E1[0] - S2[0] - K_S2E2 * (E1[1] - S2[1]))

    if SignS1 * SignE1 >= 0 or SignS2 * SignE2 >= 0:
        # Constraint 2:
        A1 = A1B1C1[5]
        C1 = A1B1C1[-6]
        B1 = A1B1C1[len(A1B1C1) // 2]

        A2 = A2B2C2[5]
        C2 = A2B2C2[-6]
        B2 = A2B2C2[len(A2B2C2) // 2]

        A1C1 = np.sqrt(np.power(A1[0] - C1[0], 2) + np.power(A1[1] - C1[1], 2))
        A2C2 = np.sqrt(np.power(A2[0] - C2[0], 2) + np.power(A2[1] - C2[1], 2))

        A1B2 = np.sqrt(np.power(A1[0] - B2[0], 2) + np.power(A1[1] - B2[1], 2))
        C1B2 = np.sqrt(np.power(C1[0] - B2[0], 2) + np.power(C1[1] - B2[1], 2))
        A2B1 = np.sqrt(np.power(A2[0] - B1[0], 2) + np.power(A2[1] - B1[1], 2))
        C2B1 = np.sqrt(np.power(C2[0] - B1[0], 2) + np.power(C2[1] - B1[1], 2))

        p1 = (A1B2 + C1B2 + A1C1) / 2
        S_A1B2C1 = np.sqrt(p1 * (p1 - A1B2) * (p1 - C1B2) * (p1 - A1C1))
        p2 = (A2B1 + C2B1 + A2C2) / 2
        S_A2B1C2 = np.sqrt(p2 * (p2 - A2B1) * (p2 - C2B1) * (p2 - A2C2))

        R1 = (A1B2 * C1B2 * A1C1) / (4 * S_A1B2C1)
        R2 = (A2B1 * C2B1 * A2C2) / (4 * S_A2B1C2)

        if abs(R1 - R2) <= T_r:
            a11 = 2 * (B2[1] - A1[1])
            b11 = 2 * (B2[0] - A1[0])
            c11 = (
                np.power(B2[1], 2)
                + np.power(B2[0], 2)
                - np.power(A1[1], 2)
                - np.power(A1[0], 2)
            )
            a12 = 2 * (C1[1] - B2[1])
            b12 = 2 * (C1[0] - B2[0])
            c12 = (
                np.power(C1[1], 2)
                + np.power(C1[0], 2)
                - np.power(B2[1], 2)
                - np.power(B2[0], 2)
            )

            a21 = 2 * (B1[1] - A2[1])
            b21 = 2 * (B1[0] - A2[0])
            c21 = (
                np.power(B1[1], 2)
                + np.power(B1[0], 2)
                - np.power(A2[1], 2)
                - np.power(A2[0], 2)
            )
            a22 = 2 * (C2[1] - B1[1])
            b22 = 2 * (C2[0] - B1[0])
            c22 = (
                np.power(C2[1], 2)
                + np.power(C2[0], 2)
                - np.power(B1[1], 2)
                - np.power(B1[0], 2)
            )

            O1_x = (a11 * c12 - a12 * c11) / (a11 * b12 - a12 * b11)
            O1_y = (c11 * b12 - c12 * b11) / (a11 * b12 - a12 * b11)

            O2_x = (a21 * c22 - a22 * c21) / (a21 * b22 - a22 * b21)
            O2_y = (c21 * b22 - c22 * b21) / (a21 * b22 - a22 * b21)

            distO1O2 = np.sqrt(np.power(O1_y - O2_y, 2) + np.power(O1_x - O2_x, 2))

            if distO1O2 <= T_o:
                R = (R1 + R2) / 2
                O_x = (O1_x + O2_x) / 2
                O_y = (O1_y + O2_y) / 2

                num1 = sum(
                    [
                        1
                        for point in A1B1C1
                        if np.abs(
                            np.sqrt(
                                np.power(point[0] - O_x, 2)
                                + np.power(point[1] - O_y, 2)
                            )
                            - R
                        )
                        <= 2
                    ]
                )
                num2 = sum(
                    [
                        1
                        for point in A2B2C2
                        if np.abs(
                            np.sqrt(
                                np.power(point[0] - O_x, 2)
                                + np.power(point[1] - O_y, 2)
                            )
                            - R
                        )
                        <= 2
                    ]
                )
                size1 = len(A1B1C1)
                size2 = len(A2B2C2)

                edgeInlier = (num1 + num2) / (size1 + size2)

                if edgeInlier >= 0.2:
                    flag = True
                    temp1_center_radius[0] = O1_x
                    temp1_center_radius[1] = O1_y
                    temp1_center_radius[2] = R1
                    temp2_center_radius[0] = O2_x
                    temp2_center_radius[1] = O2_y
                    temp2_center_radius[2] = R2

    return flag, temp1_center_radius, temp2_center_radius


def estimate_center_radius(A1B1C1, A2B2C2):
    """
    估算两组点集表示的两个弧的圆心和半径
    """
    A1 = A1B1C1[5]
    C1 = A1B1C1[-6]
    B1 = A1B1C1[len(A1B1C1) // 2]

    A2 = A2B2C2[5]
    C2 = A2B2C2[-6]
    B2 = A2B2C2[len(A2B2C2) // 2]

    A1C1 = np.sqrt(np.power(A1[0] - C1[0], 2) + np.power(A1[1] - C1[1], 2))
    A2C2 = np.sqrt(np.power(A2[0] - C2[0], 2) + np.power(A2[1] - C2[1], 2))

    A1B2 = np.sqrt(np.power(A1[0] - B2[0], 2) + np.power(A1[1] - B2[1], 2))
    C1B2 = np.sqrt(np.power(C1[0] - B2[0], 2) + np.power(C1[1] - B2[1], 2))
    A2B1 = np.sqrt(np.power(A2[0] - B1[0], 2) + np.power(A2[1] - B1[1], 2))
    C2B1 = np.sqrt(np.power(C2[0] - B1[0], 2) + np.power(C2[1] - B1[1], 2))

    p1 = (A1B2 + C1B2 + A1C1) / 2
    S_A1B2C1 = np.sqrt(p1 * (p1 - A1B2) * (p1 - C1B2) * (p1 - A1C1))
    p2 = (A2B1 + C2B1 + A2C2) / 2
    S_A2B1C2 = np.sqrt(p2 * (p2 - A2B1) * (p2 - C2B1) * (p2 - A2C2))

    R1 = (A1B2 * C1B2 * A1C1) / (4 * S_A1B2C1)
    R2 = (A2B1 * C2B1 * A2C2) / (4 * S_A2B1C2)

    a11 = 2 * (B2[1] - A1[1])
    b11 = 2 * (B2[0] - A1[0])
    c11 = (
        np.power(B2[1], 2)
        + np.power(B2[0], 2)
        - np.power(A1[1], 2)
        - np.power(A1[0], 2)
    )
    a12 = 2 * (C1[1] - B2[1])
    b12 = 2 * (C1[0] - B2[0])
    c12 = (
        np.power(C1[1], 2)
        + np.power(C1[0], 2)
        - np.power(B2[1], 2)
        - np.power(B2[0], 2)
    )

    a21 = 2 * (B1[1] - A2[1])
    b21 = 2 * (B1[0] - A2[0])
    c21 = (
        np.power(B1[1], 2)
        + np.power(B1[0], 2)
        - np.power(A2[1], 2)
        - np.power(A2[0], 2)
    )
    a22 = 2 * (C2[1] - B1[1])
    b22 = 2 * (C2[0] - B1[0])
    c22 = (
        np.power(C2[1], 2)
        + np.power(C2[0], 2)
        - np.power(B1[1], 2)
        - np.power(B1[0], 2)
    )

    O1_x = (a11 * c12 - a12 * c11) / (a11 * b12 - a12 * b11)
    O1_y = (c11 * b12 - c12 * b11) / (a11 * b12 - a12 * b11)

    O2_x = (a21 * c22 - a22 * c21) / (a21 * b22 - a22 * b21)
    O2_y = (c21 * b22 - c22 * b21) / (a21 * b22 - a22 * b21)

    R_A1C1A2 = (R1 + R2) / 2
    O_A1C1A2_x = (O1_x + O2_x) / 2
    O_A1C1A2_y = (O1_y + O2_y) / 2

    R_A1C1C2 = (R1 + R_A1C1A2) / 2
    O_A1C1C2_x = (O1_x + O_A1C1A2_x) / 2
    O_A1C1C2_y = (O1_y + O_A1C1A2_y) / 2

    R_A2C2A1 = (R2 + R_A1C1A2) / 2
    O_A2C2A1_x = (O2_x + O_A1C1A2_x) / 2
    O_A2C2A1_y = (O2_y + O_A1C1A2_y) / 2

    R_A2C2C1 = (R2 + R_A1C1C2) / 2
    O_A2C2C1_x = (O2_x + O_A1C1C2_x) / 2
    O_A2C2C1_y = (O2_y + O_A1C1C2_y) / 2

    tempR = [R1, R2, R_A1C1A2, R_A1C1C2, R_A2C2A1, R_A2C2C1]
    tempR.sort()
    estimateR = (tempR[2] + tempR[3]) / 2.0

    tempOx = [O1_x, O2_x, O_A1C1A2_x, O_A1C1C2_x, O_A2C2A1_x, O_A2C2C1_x]
    tempOx.sort()
    estimateO_x = (tempOx[2] + tempOx[3]) / 2.0

    tempOy = [O1_y, O2_y, O_A1C1A2_y, O_A1C1C2_y, O_A2C2A1_y, O_A2C2C1_y]
    tempOy.sort()
    estimateO_y = (tempOy[2] + tempOy[3]) / 2.0

    estimateO = (estimateO_x, estimateO_y)

    return estimateR, estimateO


def estimate_single_center_radius(A1B1C1):
    """
    估算单个弧的圆心和半径
    """
    A1 = A1B1C1[5]
    C1 = A1B1C1[-6]
    B1 = A1B1C1[len(A1B1C1) // 2]
    D11 = A1B1C1[5 + (len(A1B1C1) - 5) // 2]
    D12 = A1B1C1[len(A1B1C1) // 2 + (len(A1B1C1) - len(A1B1C1) // 2) // 2]

    R_A1B1C1, O_A1B1C1 = com_cir_center_radius(A1, B1, C1)
    if R_A1B1C1 is None or O_A1B1C1 is None:
        return None, None

    R_A1D11C1, O_A1D11C1 = com_cir_center_radius(A1, D11, C1)
    if R_A1D11C1 is None or O_A1D11C1 is None:
        return None, None

    R_A1D12C1, O_A1D12C1 = com_cir_center_radius(A1, D12, C1)
    if R_A1D12C1 is None or O_A1D12C1 is None:
        return None, None

    tempR = [R_A1B1C1, R_A1D11C1, R_A1D12C1]
    tempR.sort()
    estimateR = tempR[1]

    tempOX = [O_A1B1C1[0], O_A1D11C1[0], O_A1D12C1[0]]
    tempOX.sort()
    estimateOX = tempOX[1]

    tempOY = [O_A1B1C1[1], O_A1D11C1[1], O_A1D12C1[1]]
    tempOY.sort()
    estimateOY = tempOY[1]

    estimateO = (estimateOX, estimateOY)

    return estimateR, estimateO


def estimate_closed_center_radius(A1B1C1):
    """
    通过一个闭合圆上的点集估算圆心和半径
    """
    ptNum = len(A1B1C1)
    A = A1B1C1[ptNum // 5]
    B = A1B1C1[2 * ptNum // 5]
    C = A1B1C1[3 * ptNum // 5]
    D = A1B1C1[4 * ptNum // 5]
    E = A1B1C1[-7]

    R_ACD, O_ACD = com_cir_center_radius(A, C, D)
    R_ACE, O_ACE = com_cir_center_radius(A, C, E)
    R_ABD, O_ABD = com_cir_center_radius(A, B, D)
    R_BCE, O_BCE = com_cir_center_radius(B, C, E)
    R_DBE, O_DBE = com_cir_center_radius(D, B, E)

    tempR = [R_ACD, R_ACE, R_ABD, R_BCE, R_DBE]
    tempR.sort()
    estimateR = tempR[2]

    tempOX = [O_ACD[0], O_ACE[0], O_ABD[0], O_BCE[0], O_DBE[0]]
    tempOX.sort()
    estimateOX = tempOX[2]

    tempOY = [O_ACD[1], O_ACE[1], O_ABD[1], O_BCE[1], O_DBE[1]]
    tempOY.sort()
    estimateOY = tempOY[2]

    estimateO = (estimateOX, estimateOY)

    return estimateR, estimateO


def pinv_eigen_based(origin, er=1e-6):
    """
    基于SVD计算矩阵的伪逆
    """
    try:
        pinv_matrix = pinv(origin, rcond=er)
        return pinv_matrix
    except ValueError:
        # If the input matrix contains invalid values, return None
        pass


def refine_center_radius(circle_pt, center_radius):
    """
    优化圆
    """
    pt_num = len(circle_pt)
    if not center_radius or len(center_radius) < 3:
        return None

    A = np.zeros((pt_num, 3), dtype=float)
    E = np.zeros((pt_num, 1), dtype=float)
    circle_center = np.array([center_radius[0], center_radius[1]])
    circle_r = center_radius[2]

    for i in range(pt_num):
        A[i, 0] = circle_pt[i][0] - circle_center[0]
        A[i, 1] = circle_pt[i][1] - circle_center[1]
        A[i, 2] = circle_r
        E[i] = (A[i, 0] ** 2) + (A[i, 1] ** 2) - (A[i, 2] ** 2)

    pinv_A = pinv_eigen_based(A)
    if pinv_A is None:
        return None
    center_radius_refinement = np.zeros(3)
    center_radius_refinement[0] = center_radius[0] - (-0.5 * np.dot(pinv_A, E))[0]
    center_radius_refinement[1] = center_radius[1] - (-0.5 * np.dot(pinv_A, E))[1]
    center_radius_refinement[2] = center_radius[2] - (-0.5 * np.dot(pinv_A, E))[2]

    return center_radius_refinement


class GroupArcs:
    def __init__(self):
        self.arcs_from_same_circles = []  # 存储来自相同圆的弧
        self.arcs_start_mid_end = []  # 存储弧的起点、中点和终点
        self.record_or = []  # 记录估计的圆心和半径


def co_circle_group_arcs(
    edgelist: List[List[Tuple[float, float]]], T_o: int, T_r: int
) -> GroupArcs:
    """
    对圆弧进行分组
    """
    arcs = GroupArcs()
    vec = [(0, 0)]
    for i in range(len(edgelist)):
        leng = len(edgelist[i])
        if leng == 1:
            continue
        CirPt = []
        outEdgeList = [edgelist[i]]
        groupedOR = []
        outThreePt = [edgelist[i][0], edgelist[i][-1], edgelist[i][leng // 2]]
        CenterRadius = []
        for j in range(len(edgelist)):
            if j == i or len(edgelist[j]) == 1:
                continue
            else:
                flag = False
                pass_count = 0
                temp1_center_radius = []
                temp2_center_radius = []
                for k in range(len(outEdgeList)):
                    result = two_arcs_center_radius(
                        outEdgeList[k], edgelist[j], T_o, T_r
                    )
                    if result[0]:
                        pass_count += 1
                        CenterRadius.extend(result[1:])
                if pass_count == len(outEdgeList):
                    outEdgeList.append(edgelist[j])
                    start2 = edgelist[j][0]
                    end2 = edgelist[j][-1]
                    mid2 = edgelist[j][len(edgelist[j]) // 2]
                    outThreePt.extend([start2, end2, mid2])
                    edgelist[j] = vec
        for l1 in range(len(outEdgeList)):
            for l2 in range(len(outEdgeList[l1])):
                CirPt.append(outEdgeList[l1][l2])
        if len(outEdgeList) == 1:
            single_R, single_O = estimate_single_center_radius(edgelist[i])
            if single_O is not None:
                groupedOR = [single_O[0], single_O[1], single_R]
        if len(outEdgeList) == 2:
            two_R, two_O = estimate_center_radius(outEdgeList[0], outEdgeList[1])
            if two_O is not None:
                groupedOR = [two_O[0], two_O[1], two_R]
        if len(outEdgeList) > 2:
            temp_r = []
            temp_x = []
            temp_y = []
            for item in CenterRadius:
                temp_x.append(item[0])
                temp_y.append(item[1])
                temp_r.append(item[2])
            temp_r.sort()
            temp_x.sort()
            temp_y.sort()
            num_center_radius = len(CenterRadius)
            if num_center_radius % 2 == 0:
                MoreR = (
                    temp_r[num_center_radius // 2]
                    + temp_r[(num_center_radius // 2) + 1]
                ) / 2.0
                MoreO = [
                    (
                        temp_x[num_center_radius // 2]
                        + temp_x[(num_center_radius // 2) + 1]
                    )
                    / 2,
                    (
                        temp_y[num_center_radius // 2]
                        + temp_y[(num_center_radius // 2) + 1]
                    )
                    / 2,
                ]
            else:
                MoreR = temp_r[(num_center_radius + 1) // 2.0]
                MoreO = [
                    temp_x[(num_center_radius + 1) // 2.0],
                    temp_y[(num_center_radius + 1) // 2.0],
                ]
            if MoreO is not None:
                groupedOR = [MoreO[0], MoreO[1], MoreR]
        center_radius_refinement = refine_center_radius(CirPt, groupedOR)
        edgelist[i] = vec
        arcs.arcs_from_same_circles.append(CirPt)
        arcs.arcs_start_mid_end.append(outThreePt)
        arcs.record_or.append(center_radius_refinement)
    return arcs


class Circles:
    def __init__(self):
        self.xc = 0.0
        self.yc = 0.0
        self.r = 0.0
        self.inlierRatio = 0.0


import numpy as np


def circle_verify(x, y, N, stEdMid, O, R):
    pe = 0
    angle = 0

    pxc = O[0]
    pyc = O[1]

    if np.isinf(R) or np.isnan(R) or (R <= 0):
        return False
    else:
        DT = []
        D1 = []
        D2 = []
        for j in range(N - 1):
            L = np.sqrt(np.power(x[j] - x[j + 1], 2) + np.power(y[j] - y[j + 1], 2))
            dx = (x[j] - x[j + 1]) / L
            dy = (y[j] - y[j + 1]) / L
            D1.append([dx, dy])
            D2.append([dx, dy])

        LL = np.sqrt(np.power(x[-1] - x[0], 2) + np.power(y[-1] - y[0], 2))
        end2start = [(x[-1] - x[0]) / LL, (y[-1] - y[0]) / LL]
        D1.append(end2start)
        D2.append(end2start)

        D = []
        Nom = []
        for i in range(N):
            D.append([D1[i][0] + D2[i][0], D1[i][1] + D2[i][1]])
            LL1 = np.sqrt(np.power(D[i][0], 2) + np.power(D[i][1], 2))
            Nom.append([-D[i][1] / LL1, D[i][0] / LL1])

        inlierNum = 0
        for i in range(N):
            dx = x[i] - pxc
            dy = y[i] - pyc
            distP2O = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
            d = np.abs(distP2O - R)
            if d <= 2:
                theta = np.arctan2(dy, dx)
                circleNormal = [np.cos(theta), np.sin(theta)]
                cosPtCirNormal2 = np.dot(Nom[i], circleNormal)
                if np.abs(cosPtCirNormal2) >= np.cos(22.5 / 180 * np.pi):
                    inlierNum += 1

        inlierEdgeRatio = inlierNum / N
        inlierRatio = 0
        spanAngle = 0

        if inlierEdgeRatio >= 0.2:
            inlierRatio = inlierNum / (2 * np.pi * R)

        pe = inlierRatio
        angle = spanAngle

    return True, pe, angle


def circle_estimate_grouped_arcs(
    groupedArcs, recordOR, groupedArcsThreePt, T_inlier, T_angle
):
    addCircles = []
    for i, arcs in enumerate(groupedArcs):
        fitCircle = Circles()
        X, Y = [], []  # Point coordinates
        stEdMid = groupedArcsThreePt[i]
        record = recordOR[i]
        if record is None:
            continue  # Skip if record is None
        groupedR = record[2]
        groupedO = (record[0], record[1])
        for arc in arcs:
            X.append(arc[0])
            Y.append(arc[1])

        # Fit
        result = circle_verify(X, Y, len(X), stEdMid, groupedO, groupedR)
        if not isinstance(result, tuple):
            print(f"Unexpected result type from circle_verify: {result}")
            continue
        success, inlierRatio, spanAngle = result

        # Inlier verification
        if success and inlierRatio >= T_inlier and groupedR >= 5:
            fitCircle.xc = groupedO[0]
            fitCircle.yc = groupedO[1]
            fitCircle.r = groupedR
            fitCircle.inlierRatio = inlierRatio

            addCircles.append(fitCircle)

    return addCircles


def circle_estimate_closed_arcs(closed_arcs, T_inlier_closed):
    add_circles = []

    for i, closed_arc in enumerate(closed_arcs):
        X, Y = [], []
        three_pt = []
        closed_r = 0
        closed_o = (0, 0)  # Initialize center

        # Estimate center and radius
        for point in closed_arc:
            Y.append(point[1])
            X.append(point[0])

        # Fit circle
        result = circle_verify(X, Y, len(X), three_pt, closed_o, closed_r)

        # Check if result is valid
        if isinstance(result, tuple) and len(result) >= 3:
            success, inlier_ratio, span_angle = result

            # Inlier verification
            if success and inlier_ratio >= 0.5 and closed_r >= 4:
                fit_circle = Circles(closed_o[0], closed_o[1], closed_r, inlier_ratio)
                add_circles.append(fit_circle)

    return add_circles


def cluster_circles(total_circles):
    rep_circles = []

    while total_circles:
        sim_circles = []
        circle1 = total_circles.pop(0)

        it = 0
        while it < len(total_circles):
            circle2 = total_circles[it]
            dis_circles = np.sqrt(
                (circle1.xc - circle2.xc) ** 2
                + (circle1.yc - circle2.yc) ** 2
                + (circle1.r - circle2.r) ** 2
            )
            if dis_circles <= 5:
                sim_circles.append(circle2)
                total_circles.pop(it)
            else:
                it += 1

        if len(sim_circles) > 1:
            sim_circles.sort(key=lambda x: x.inlierRatio, reverse=True)

        rep_circles.append(sim_circles[0] if sim_circles else circle1)
        print(
            "Final inlier ratio:",
            sim_circles[0].inlierRatio if sim_circles else circle1.inlierRatio,
        )

    return rep_circles



