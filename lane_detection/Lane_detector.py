#!/usr/bin/env python3

import rospy
import cv2
import math
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from ackermann_msgs.msg import AckermannDriveStamped

class LaneFollow:
    def __init__(self):
        self.br = CvBridge()
        # self.image_sub = rospy.Subscriber('/oak/rgb/image_rect', Image, self.image_callback)
        self.image_sub = rospy.Subscriber('/oak/rgb/image_rect_color/compressed', CompressedImage, self.image_callback)
        self.ack_pub = rospy.Publisher('/high_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
        self.debug_publisher1 = rospy.Publisher('/debugging_image1', Image, queue_size=10)
        self.debug_publisher2 = rospy.Publisher('/debugging_image2', Image, queue_size=10)

        # ROI 파라미터
        self.roi_x_l = rospy.get_param('~roi_x_l', 0)
        self.roi_x_h = rospy.get_param('~roi_x_h', 1280)
        self.roi_y_l = rospy.get_param('~roi_y_l', 380)  # ROI 영역을 아래로 조정
        self.roi_y_h = rospy.get_param('~roi_y_h', 720)

        # 차선 검출 파라미터
        self.white_lane_low = np.array([0, 0, 180])  # Saturation 낮추고, Value 상향
        self.white_lane_high = np.array([180, 45, 210]) # Hue 범위 확대
        self.brightness_threshold = 240  # 반사광 제거를 위한 밝기 임계값
        self.min_contour_area = 2000     # 최소 컨투어 영역
        self.min_line_length = 30      # 최소 선 길이
        self.max_line_width = 100        # 최대 선 너비
        self.min_aspect_ratio = 2.5   # 최소 종횡비
        self.max_roughness = 1.03

        # 주행 관련 변수
        self.current_center = 640
        self.prev_centers = []
        self.center_history_size = 3
        self.guide_point_distance_x = rospy.get_param('~guide_point_distance_x', 0.3)
        self.lane_offset = rospy.get_param('~lane_offset', 375)
        
        # 디버깅
        self.debug_sequence = rospy.get_param('~debug_image_num', 1)
        self.kernel = np.ones((3, 3), np.uint8)
        
        # 원근 변환 매트릭스
        self.src_points = np.float32([
            [self.roi_x_l, self.roi_y_l],
            [self.roi_x_h, self.roi_y_l],
            [self.roi_x_l, self.roi_y_h],
            [self.roi_x_h, self.roi_y_h]
        ])

        self.dst_points = np.float32([
            [self.roi_x_l, self.roi_y_l],
            [self.roi_x_h, self.roi_y_l],
            [self.roi_x_l + 360, self.roi_y_h],
            [self.roi_x_h - 360, self.roi_y_h]
        ])

        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inv_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
        self.image_ = None

    def image_callback(self, msg):
        self.image_ = self.br.compressed_imgmsg_to_cv2(msg, 'bgr8')

    def preprocess_image(self, image):
        """반사광 제거를 위한 강화된 전처리"""
        # 가우시안 블러
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # HSV 변환
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # 반사광 영역 마스크 생성
        reflection_mask = v > 220  # 매우 밝은 영역을 반사광으로 간주
        
        # 반사광 영역의 채도(S) 높이기 (노란색 차선은 채도가 높음)
        # s[reflection_mask] = s[reflection_mask] * 1.2   #1.5
        # s = np.clip(s, 0, 255).astype(np.uint8)
        
        # 반사광 영역의 밝기(V) 감소
        v[reflection_mask] = v[reflection_mask] * 0.75   #0.5
        v = np.clip(v, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    def filter_contours(self, contours):
        """반사광 제거를 위한 강화된 컨투어 필터링 (외곽선의 roughness 적용)"""
        filtered_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue

            # 컨투어 영역의 평균 밝기 및 채도 계산
            mask = np.zeros_like(self.roi_[:, :, 0])
            cv2.drawContours(mask, [contour], 0, 255, -1)

            # 형태 분석
            rect = cv2.minAreaRect(contour)
            height = max(rect[1])
            width = min(rect[1])
            
            if height < self.min_line_length:
                continue

            if width > self.max_line_width:
                continue
            
            if height / width < self.min_aspect_ratio:  # 종횡비 조건 강화
                continue
            
            # 면적 대비 둘레 길이 검사 (반사광은 보통 불규칙한 형태)
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.6:  # 너무 원형에 가까운 형태 제외
                continue

            epsilon = 0.01 * perimeter  # 허용 오차, 필요에 따라 조정 가능
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_perimeter = cv2.arcLength(approx, True)
            
            # 원래 둘레와 단순화된 둘레의 비율을 roughness factor로 사용
            roughness = perimeter / approx_perimeter if approx_perimeter > 0 else 1
            if roughness > self.max_roughness:
                continue

            filtered_contours.append(contour)
        
        return filtered_contours

    

    def run(self, deaccel_flag=1):
        """메인 실행 함수"""
        if self.image_ is None:
            return
        
        # 원근 변환 및 ROI 추출
        self.warp_ = cv2.warpPerspective(self.image_, self.matrix, (self.image_.shape[1], self.image_.shape[0]))
        self.roi_ = self.warp_[self.roi_y_l:self.roi_y_h, self.roi_x_l:self.roi_x_h]
        
        # 반사광 억제 전처리
        self.roi_ = self.preprocess_image(self.roi_)
        
        # HSV 변환 및 마스킹
        hsv = cv2.cvtColor(self.roi_, cv2.COLOR_BGR2HSV)
        self.mask_white = cv2.inRange(hsv, self.white_lane_low, self.white_lane_high)
        
        
        # run 함수 내에서
        # HSV 변환 및 마스킹 후
        self.mask_white = cv2.morphologyEx(self.mask_white, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        self.mask_white = cv2.morphologyEx(self.mask_white, cv2.MORPH_OPEN, self.kernel)
        self.mask_white = cv2.dilate(self.mask_white, self.kernel, iterations=1)  # 차선 두께 강화

        # 컨투어 검출 및 필터링
        contours, _ = cv2.findContours(self.mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        filtered_contours = self.filter_contours(contours)

        # 차선 분류
        left_lines = []
        right_lines = []
        self.img_center = self.roi_.shape[1] // 2

        for contour in filtered_contours:
            [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            slope = vy/vx if vx != 0 else float('inf')
            
            # 너무 수평인 선 제외
            if abs(slope) < 0.4:
                continue
            
            # 차선의 중심점 계산
            topy = 0
            bottomy = self.roi_.shape[0]
            topx = int((topy - y0) * vx / vy + x0)
            bottomx = int((bottomy - y0) * vx / vy + x0)
            centerx = (topx + bottomx) // 2
                
            if bottomx < self.img_center:
                left_lines.append(centerx)
            else:
                right_lines.append(centerx)
            
            # 디버깅용 시각화
            cv2.drawContours(self.roi_, [contour], 0, (0, 255, 0), 2)
            cv2.line(self.roi_, (topx, topy), (bottomx, bottomy), (0, 0, 255), 2)

        # 가이드 포인트 계산
        left_lines.sort()
        right_lines.sort()
        guide_center = None
        
        if left_lines and right_lines:
            guide_center = (left_lines[-1] + right_lines[0]) // 2
        elif len(left_lines) > 1:
            guide_center = (left_lines[-2] + left_lines[-1]) // 2 if right_lines else left_lines[0] + self.lane_offset
        elif len(right_lines) > 1:
            guide_center = (right_lines[0] + right_lines[1]) // 2 if left_lines else right_lines[0] - self.lane_offset
        elif left_lines:
            guide_center = left_lines[0] + self.lane_offset
        elif right_lines:
            guide_center = right_lines[0] - self.lane_offset
            
            
        # if len(left_lines) + len(right_lines) == 0:
        #     # 이전 프레임의 중심점 유지하되 약간의 보정
        #     if abs(self.current_center - self.img_center) > 50:  # 중앙에서 많이 벗어난 경우
        #         self.current_center = int(0.8 * self.current_center + 0.2 * self.img_center)  # 중앙으로 천천히 복귀

            
        if guide_center is not None:
            self.current_center = guide_center
        
        cv2.circle(self.roi_, (self.current_center, 200), 30, (0, 255, 0), thickness=-1)
        self.calculate_guiding_position(self.current_center, deaccel_flag)
        
        #디버깅 이미지 발행
        if self.debug_sequence == 1:
            self.debug_publisher1.publish(self.br.cv2_to_imgmsg(self.mask_white, 'mono8'))
            self.debug_publisher2.publish(self.br.cv2_to_imgmsg(self.roi_, 'bgr8'))
            
    

if __name__ == '__main__':
    try:
        rospy.init_node('Lane_Follow')
        lane_follow = LaneFollow()
        rospy.sleep(1)

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            lane_follow.run()
            rate.sleep()

    except rospy.ROSInterruptException:
        pass