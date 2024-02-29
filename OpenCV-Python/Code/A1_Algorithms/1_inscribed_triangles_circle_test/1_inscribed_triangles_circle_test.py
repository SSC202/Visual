import cv2
import numpy as np
import random

import RDP
import CloseEdge
import RejectSharp
import DetInflexPt
import CircleTool

# 全局定义段
# 1. 饱和度增强定义
# 调节通道强度
lutEqual = np.array([i for i in range(256)]).astype("uint8")
lutRaisen = np.array([int(102 + 0.6 * i) for i in range(256)]).astype("uint8")
# 调节饱和度
lutSRaisen = np.dstack((lutEqual, lutRaisen, lutEqual))  # Saturation raisen
# 2. 掩膜阈值定义
lower_red = np.array([116, 156, 0])
upper_red = np.array([179, 255, 255])
# 3. 结构元素定义
kernel = np.ones((3, 3), np.uint8)
# 4. 摄像头定义
cap = cv2.VideoCapture(1)
cap.set(10, 2)
# 5. 边缘检测定义
ed = cv2.ximgproc.createEdgeDrawing()  # 创建ED对象
# 6. 圆检测器定义
sharp_angle = 40
T_l = 20
T_ratio = 0.001
T_o = 5
# 5 10 15 20 25
T_r = 5
# 5 10 15 20 25
T_inlier = 0.3
# 0.3 0.35 0.4 0.45 0.5 (the larger the more strict)
T_angle = 2.0
T_inlier_closed = 0.5

while cap.isOpened() == True:
    ret, frame = cap.read()
    if ret == True:
        # 1. 饱和度增强(可选)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # 色彩空间转换, RGB->HSV
        blendSRaisen = cv2.LUT(hsv, lutSRaisen)  # 饱和度增大

        # 2. 掩膜处理
        red_mask = cv2.inRange(blendSRaisen, lower_red, upper_red)
        red_frame = cv2.bitwise_and(frame, frame, mask=red_mask)

        # 3. 滤波
        red_frame = cv2.medianBlur(red_frame, 3)

        # 4. 转换为灰度图
        red_gray = cv2.cvtColor(red_frame, cv2.COLOR_BGR2GRAY)

        # 5. 使用EDPF算法进行边缘提取
        ed.detectEdges(red_gray)
        red_edge = ed.getEdgeImage()
        red_edge_segments = ed.getSegments()

        # 6. 删除像素数小于16的边缘
        red_edge_list = []
        for segment in red_edge_segments:
            if len(segment) >= 16:
                red_edge_list.append(segment)

        # 7. 提取闭合边缘
        red_closed_and_not_closed_edges = CloseEdge.extract_closed_edges(red_edge_list)
        red_closed_edge_list = red_closed_and_not_closed_edges.closed_edges

        # 8. 使用Ramer-Douglas-Peucker方法近似边缘段为线段
        red_seg_list = []
        for edge in red_edge_list:
            red_seg_temp = []  # 创建一个空的临时列表用于存储近似后的线段点
            RDP.ramer_douglas_peucker(
                edge, 2.5, red_seg_temp
            )  # 使用Ramer-Douglas-Peucker算法近似边缘段为线段，阈值为2.5
            red_seg_list.append(red_seg_temp)  # 将近似后的线段点添加到列表中

        # 9. 删去锐角轮廓
        red_new_seg_edge_list = RejectSharp.reject_sharp_turn(
            red_edge_list, red_seg_list, sharp_angle
        )
        red_new_seg_list = red_new_seg_edge_list.new_segList
        red_new_edge_list = red_new_seg_edge_list.new_edgeList

        # 检测段
        # for j in range(len(red_new_seg_list)):  # 遍历新的边缘段列表
        #     r = random.randint(0, 255)  # 生成随机颜色值
        #     g = random.randint(0, 255)
        #     b = random.randint(0, 255)
        #     color_sharp_turn = (b, g, r)  # 创建边缘段的随机颜色
        #     for jj2 in range(len(red_new_edge_list[j]) - 1):  # 遍历当前边缘段的每个点
        #         # 用随机颜色绘制线段
        #         cv2.line(frame, red_new_edge_list[j][jj2], red_new_edge_list[j][jj2 + 1], color_sharp_turn, 2)

        # 10. 检测拐点
        red_new_seg_edge_list_after_inflexion = DetInflexPt.detectInflexPt(
            red_new_edge_list, red_new_seg_list
        )

        red_new_seg_list_after_inflexion = (
            red_new_seg_edge_list_after_inflexion.new_segList
        )
        red_new_edge_list_after_inflexion = (
            red_new_seg_edge_list_after_inflexion.new_edgeList
        )

        # 11. 删除短边缘段或接近线段
        it = 0
        while it < len(red_new_edge_list_after_inflexion):
            edge_st = (
                red_new_edge_list_after_inflexion[it][0][0],
                red_new_edge_list_after_inflexion[it][0][1],
            )
            edge_ed = (
                red_new_edge_list_after_inflexion[it][-1][0],
                red_new_edge_list_after_inflexion[it][-1][1],
            )
            mid_index = len(red_new_edge_list_after_inflexion[it]) // 2
            edge_mid = (
                red_new_edge_list_after_inflexion[it][mid_index][0],
                red_new_edge_list_after_inflexion[it][mid_index][1],
            )

            dist_st_ed = np.sqrt(
                (edge_st[0] - edge_ed[0]) ** 2 + (edge_st[1] - edge_ed[1]) ** 2
            )
            dist_st_mid = np.sqrt(
                (edge_st[0] - edge_mid[0]) ** 2 + (edge_st[1] - edge_mid[1]) ** 2
            )
            dist_mid_ed = np.sqrt(
                (edge_ed[0] - edge_mid[0]) ** 2 + (edge_ed[1] - edge_mid[1]) ** 2
            )
            dist_difference = abs((dist_st_mid + dist_mid_ed) - dist_st_ed)

            if len(
                red_new_edge_list_after_inflexion[it]
            ) <= T_l or dist_difference <= T_ratio * (dist_st_mid + dist_mid_ed):
                del red_new_edge_list_after_inflexion[it]
            else:
                it += 1

        # # 检测段
        # for j in range(len(red_new_edge_list_after_inflexion)):
        #     # 生成随机颜色
        #     r = np.random.randint(256)
        #     g = np.random.randint(256)
        #     b = np.random.randint(256)
        #     color_after_delete_line_pt = (b, g, r)

        #     for jj2 in range(len(red_new_edge_list_after_inflexion[j]) - 1):
        #         # 使用随机颜色绘制线段
        #         cv2.line(frame, red_new_edge_list_after_inflexion[j][jj2], red_new_edge_list_after_inflexion[j][jj2 + 1], color_after_delete_line_pt, 2)

        # 12. 检测闭合性
        red_closed_and_not_closed_Edges_1 = CloseEdge.extract_closed_edges(
            red_new_edge_list_after_inflexion
        )
        red_closed_edge_list_1 = red_closed_and_not_closed_Edges_1.closed_edges
        red_not_closed_edge_list_1 = red_closed_and_not_closed_Edges_1.not_closed_edges

        # 检测段
        # for closedEdge in red_closed_edge_list_1:
        #     colorClosedEdges = (
        #         np.random.randint(0, 256),
        #         np.random.randint(0, 256),
        #         np.random.randint(0, 256),
        #     )
        #     for jj in range(len(closedEdge) - 1):
        #         cv2.line(
        #             frame, closedEdge[jj], closedEdge[jj + 1], colorClosedEdges, 2
        #         )

        # 13. 分组
        red_sorted_edge_list = CircleTool.sort_edge_list(red_not_closed_edge_list_1)
        red_arcs = CircleTool.co_circle_group_arcs(red_sorted_edge_list, T_o, T_r)
        red_groupedArcs = red_arcs.arcs_from_same_circles
        red_groupedArcsThreePt = red_arcs.arcs_start_mid_end
        red_groupedOR = red_arcs.record_or
        # 14. 估计和验证
        red_grouped_circles = CircleTool.circle_estimate_grouped_arcs(
            red_groupedArcs, red_groupedOR, red_groupedArcsThreePt, T_inlier, T_angle
        )  # 拟合分组的弧
        # 封闭弧段
        for ite in red_closed_edge_list_1:
            red_closed_edge_list.append(ite)
        red_closed_circles = CircleTool.circle_estimate_closed_arcs(
            red_closed_edge_list, T_inlier_closed
        )  # 拟合封闭边缘
        red_totalCircles = []
        if red_grouped_circles:
            red_totalCircles.extend(red_grouped_circles)
        if red_closed_circles:
            red_totalCircles.extend(red_closed_circles)
        preCircles = CircleTool.cluster_circles(red_totalCircles)

        # 检测
        for circle in preCircles:
            xc, yc, r = int(circle.xc), int(circle.yc), int(circle.r)
            cv2.circle(frame, (xc, yc), r, (0, 255, 0), 2)
        cv2.imshow("res", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()
