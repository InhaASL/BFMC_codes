#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from std_msgs.msg import Header

class PolyfitLaneStopLineDetector:
    def __init__(self):
        self.bridge = CvBridge()
        
        rospy.init_node("polyfit_lane_stopline_node", anonymous=True)
        self.image_sub = rospy.Subscriber("/d455/infra1/image_rect_raw/compressed", 
                                        CompressedImage, self.image_callback, queue_size=1)
        self.image_pub = rospy.Publisher("/lane_stopline/debug_image", Image, queue_size=1)
        self.roi_y_l = 300
        self.roi_padding = 190
        self.init_flag = True

        self.src_points = None
        self.dst_points = None
        self.matrix = None
        self.inv_matrix = None
        self.image_height = None
        self.image_width = None

    def set_perspective_param(self, image):
        """Bird's Eye View를 위한 Perspective transform 파라미터 설정"""

        self.image_height, self.image_width = image.shape
        
        self.src_points = np.float32([
            [0, self.roi_y_l-1], 
            [self.image_width, self.roi_y_l-1],
            [0, self.image_height - 1],   
            [self.image_width - 1, self.image_height - 1] 
        ])

        self.dst_points = np.float32([
            [-self.roi_padding, self.roi_y_l],                          
            [self.image_width + self.roi_padding - 1, self.roi_y_l],   
            [self.roi_padding, self.image_height - 1],                  
            [self.image_width - self.roi_padding - 1, self.image_height - 1]  
        ])

        self.matrix = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        self.inv_matrix = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
    
    def preprocess_image(self, image):
        if image is None or image.size == 0:
            return None, None
    
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 80)
        
        # Bird's Eye View에서는 관심 영역만 처리
        roi_mask = np.zeros_like(edges)
        roi_mask[self.roi_y_l:, :] = 255
        edges = cv2.bitwise_and(edges, roi_mask)
        
        return image, edges
    
    def extract_contour_points(self, contour):
        """컨투어에서 점들을 추출하여 분석용 좌표 리스트 반환"""
        points = []
        for point in contour:
            x, y = point[0]
            points.append([x, y])
        return np.array(points)
    
    def analyze_with_polyfit(self, points, degree=2):
        """polyfit을 사용하여 점들의 방향성 분석"""
        if len(points) < degree + 1:
            return (None, float('inf')), (None, float('inf'))
        
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # x에 대한 y의 다항식 피팅 (수직선 감지용)
        try:
            poly_coeff_xy = np.polyfit(x_coords, y_coords, degree)
            poly_xy = np.poly1d(poly_coeff_xy)
            
            # 피팅 오차 계산
            y_pred = poly_xy(x_coords)
            error_xy = np.mean((y_coords - y_pred) ** 2)
        except (np.RankWarning, np.linalg.LinAlgError):
            poly_coeff_xy, error_xy = None, float('inf')
        
        # y에 대한 x의 다항식 피팅 (수평선 감지용)
        try:
            poly_coeff_yx = np.polyfit(y_coords, x_coords, degree)
            poly_yx = np.poly1d(poly_coeff_yx)
            
            # 피팅 오차 계산
            x_pred = poly_yx(y_coords)
            error_yx = np.mean((x_coords - x_pred) ** 2)
        except (np.RankWarning, np.linalg.LinAlgError):
            poly_coeff_yx, error_yx = None, float('inf')
        
        return (poly_coeff_xy, error_xy), (poly_coeff_yx, error_yx)
    
    def classify_line_type(self, points, xy_fit, yx_fit):
        """polyfit 결과를 바탕으로 차선(수직) 또는 정지선(수평) 분류"""
        (poly_xy, error_xy), (poly_yx, error_yx) = xy_fit, yx_fit
        
        # 피팅 오차가 너무 크면 제외
        if error_xy == float('inf') and error_yx == float('inf'):
            return "unknown", None
        
        # 기울기 분석을 위한 추가 검증
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        # 종횡비를 통한 1차 분류
        aspect_ratio = x_range / (y_range + 1e-6)
        
        # 임계값 조정 및 더 엄격한 분류
        if aspect_ratio > 2.5:  # 가로가 세로보다 2.5배 이상 길면 정지선 후보
            if error_yx < error_xy and error_yx < 100:  # 오차 임계값 추가
                return "stopline", poly_yx
            else:
                return "unknown", None
        elif aspect_ratio < 0.4:  # 세로가 가로보다 2.5배 이상 길면 차선 후보
            if error_xy < error_yx and error_xy < 100:  # 오차 임계값 추가
                return "lane", poly_xy
            else:
                return "unknown", None
        else:
            # 애매한 경우 더 엄격하게 판단
            min_error = min(error_xy, error_yx)
            if min_error > 50:  # 오차가 너무 크면 제외
                return "unknown", None
            
            if error_xy < error_yx:
                return "lane", poly_xy
            else:
                return "stopline", poly_yx
    
    def smooth_contour_with_polyfit(self, points, line_type, poly_coeff):
        """polyfit 결과를 바탕으로 부드러운 곡선 생성"""
        if poly_coeff is None or len(points) == 0:
            return points
        
        try:
            if line_type == "lane":
                # 차선: x를 기준으로 y 예측
                x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
                x_smooth = np.linspace(x_min, x_max, min(50, len(points)))
                poly_func = np.poly1d(poly_coeff)
                y_smooth = poly_func(x_smooth)
                
                # 유효한 범위 내의 점들만 선택
                valid_mask = (y_smooth >= 0) & (y_smooth < self.image_height)
                x_smooth = x_smooth[valid_mask]
                y_smooth = y_smooth[valid_mask]
                
                smooth_points = np.column_stack([x_smooth, y_smooth])
            else:
                # 정지선: y를 기준으로 x 예측
                y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
                y_smooth = np.linspace(y_min, y_max, min(50, len(points)))
                poly_func = np.poly1d(poly_coeff)
                x_smooth = poly_func(y_smooth)
                
                # 유효한 범위 내의 점들만 선택
                valid_mask = (x_smooth >= 0) & (x_smooth < self.image_width)
                x_smooth = x_smooth[valid_mask]
                y_smooth = y_smooth[valid_mask]
                
                smooth_points = np.column_stack([x_smooth, y_smooth])
            
            return smooth_points.astype(int)
        except Exception as e:
            rospy.logwarn(f"부드러운 곡선 생성 실패: {e}")
            return points
    
    def draw_results(self, roi, analysis_results):
        """분석 결과를 Bird's Eye View에서 시각화"""
        if roi is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Bird's Eye View 결과를 컬러로 변환
        if len(roi.shape) == 2:
            result = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        else:
            result = roi.copy()
        
        for line_type, smooth_points, original_points in analysis_results:
            if line_type == "lane":
                color = (255, 0, 0)  # 차선은 파란색
            elif line_type == "stopline":
                color = (0, 0, 255)  # 정지선은 빨간색
            else:
                color = (0, 255, 0)  # 미분류는 초록색
            
            # 원본 점들을 작은 원으로 표시
            for point in original_points:
                if 0 <= point[0] < result.shape[1] and 0 <= point[1] < result.shape[0]:
                    cv2.circle(result, tuple(point), 2, color, -1)
            
            # polyfit으로 부드럽게 만든 선 그리기
            if len(smooth_points) > 1:
                pts = smooth_points.reshape((-1, 1, 2))
                cv2.polylines(result, [pts], False, color, 3)
        
        return result
    
    def image_callback(self, msg):
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return
            
        if self.init_flag:
            self.set_perspective_param(cv_img)
            self.init_flag = False

        warp_img = cv2.warpPerspective(cv_img, self.matrix, (self.image_width, self.image_height))
            
        roi, edges = self.preprocess_image(warp_img)
            
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        analysis_results = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                continue
            
            # 컨투어 점들 추출
            points = self.extract_contour_points(contour)
            if len(points) < 5:  # 최소 점 개수 확인
                continue
            
            # polyfit 분석
            xy_fit, yx_fit = self.analyze_with_polyfit(points, degree=2)
            
            # 선 종류 분류
            line_type, poly_coeff = self.classify_line_type(points, xy_fit, yx_fit)
            
            if line_type == "unknown":
                continue
            
            # 부드러운 곡선 생성
            smooth_points = self.smooth_contour_with_polyfit(points, line_type, poly_coeff)
            
            analysis_results.append((line_type, smooth_points, points))
        
        debug_img = self.draw_results(roi, analysis_results)
        
        # 디버그 이미지 퍼블리시

        debug_msg = self.bridge.cv2_to_imgmsg(edges, encoding="passthrough")
        debug_msg.header = msg.header
        self.image_pub.publish(debug_msg)
    
        # 분석 결과 로그 출력
        lane_count = sum(1 for result in analysis_results if result[0] == "lane")
        stopline_count = sum(1 for result in analysis_results if result[0] == "stopline")
        if lane_count > 0 or stopline_count > 0:
            rospy.loginfo(f"감지된 차선: {lane_count}개, 정지선: {stopline_count}개")

if __name__ == "__main__":
    try:
        node = PolyfitLaneStopLineDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass