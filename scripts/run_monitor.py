import cv2
import os
from monitor.monitor import SocialDistancingMonitor

def main():
    monitor = SocialDistancingMonitor(weights='yolov5s.pt', imgsz=640)
    cap = cv2.VideoCapture("./Input/test4.mp4")
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs("./Output", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    perspective_out = cv2.VideoWriter("./Output/perspective.avi", fourcc, fps, (frame_width, frame_height))
    birds_eye_out = cv2.VideoWriter("./Output/birds_eye.avi", fourcc, fps, 
                                   (int(frame_width * monitor.scale_w), int(frame_height * monitor.scale_h)))
    ret, frame = cap.read()
    if not ret or not monitor.calibrate(frame):
        print("Calibration failed")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, centroid_dict, detections, _ = monitor.detect_persons(frame)
        if centroid_dict:
            _, bird_image, red_zone_ids = monitor.plot_birds_eye_view(frame, centroid_dict)
            result_frame = monitor.draw_perspective_boxes(processed_frame, detections, red_zone_ids, centroid_dict)
        else:
            result_frame, bird_image = processed_frame, None
        cv2.imshow("Social Distancing Monitor", result_frame)
        if bird_image is not None:
            cv2.imshow("Bird's Eye View", bird_image)
        perspective_out.write(result_frame)
        if bird_image is not None:
            birds_eye_out.write(bird_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    perspective_out.release()
    birds_eye_out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
