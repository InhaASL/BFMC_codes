    def calculate_guiding_position(self, guide_center, accel_flag):
        """조향각 계산 및 제어"""
        dy = guide_center - self.img_center
        dx = self.roi_.shape[0] + 300
        theta_rad = np.arctan2(dy, dx)
        
        # 급격한 변화 방지
        if self.prev_centers:
            max_center_change = 70
            prev_avg = sum(self.prev_centers) / len(self.prev_centers)
            if abs(guide_center - prev_avg) > max_center_change:
                guide_center = prev_avg + np.sign(guide_center - prev_avg) * max_center_change
                theta_rad = np.arctan2(guide_center - self.img_center, dx)
        
        self.prev_centers.append(guide_center)
        if len(self.prev_centers) > self.center_history_size:
            self.prev_centers.pop(0)
        
        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "base_link"
        msg.drive.steering_angle = -theta_rad
        print(np.rad2deg(theta_rad))
        msg.drive.speed = 1.0 if accel_flag else 2.0
        self.ack_pub.publish(msg)