import cv2
import os
from monitor.monitor import SocialDistancingMonitor

def run_demo(video_path="./Input/test4.mp4", output_dir="./Output"):
    # Initialize monitor
    monitor = SocialDistancingMonitor(weights='yolov5s.pt', imgsz=640)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define video writers
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    perspective_out = cv2.VideoWriter(os.path.join(output_dir, "perspective.avi"),
                                      fourcc, fps, (frame_width, frame_height))
    birds_eye_out = cv2.VideoWriter(os.path.join(output_dir, "birds_eye.avi"),
                                    fourcc, fps, (int(frame_width * monitor.scale_w),
                                                  int(frame_height * monitor.scale_h)))

    # Calibration
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame for calibration")
        return
    if not monitor.calibrate(frame):
        print("❌ Calibration failed — please restart and click 7 points")
        return

    print("✅ Calibration complete. Running detection...")

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect persons
        processed_frame, centroid_dict, detections, _ = monitor.detect_persons(frame)

        # If people detected
        if centroid_dict:
            _, bird_image, red_zone_ids = monitor.plot_birds_eye_view(frame, centroid_dict)
            result_frame = monitor.draw_perspective_boxes(processed_frame, detections, red_zone_ids, centroid_dict)
        else:
            result_frame = processed_frame
            bird_image = None

        # Show results
        cv2.imshow("Perspective View", result_frame)
        if bird_image is not None:
            cv2.imshow("Bird's Eye View", bird_image)

        # Write output
        perspective_out.write(result_frame)
        if bird_image is not None:
            birds_eye_out.write(bird_image)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    perspective_out.release()
    birds_eye_out.release()
    cv2.destroyAllWindows()
    print("✅ Demo finished. Results saved in:", output_dir)

if __name__ == "__main__":
    run_demo()
