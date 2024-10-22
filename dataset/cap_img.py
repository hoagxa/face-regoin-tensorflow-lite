import cv2
import time
import sys

def main(video_source=0):
    cpt = 0
    maxFrames = 30  # if you want 5 frames only.

    cap = cv2.VideoCapture(video_source)

    while cpt < maxFrames:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # frame = cv2.resize(frame, (640, 480))
        cv2.imshow("test window", frame)  # show image in window
        
        # Wait for Space key to capture the image
        key = cv2.waitKey(1)
        if key == 32:  # Space key
            cv2.imwrite("images/Diep_%d.jpg" % cpt, frame)
            print(f"Captured image {cpt + 1}")
            cpt += 1
        
        if key & 0xFF == 27:  # ESC key to exit
            break
        time.sleep(0.1)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if a video file was provided as a command line argument
    if len(sys.argv) > 1:
        video_source = sys.argv[1]  # Use the video file path as the source
    else:
        video_source = 0  # Default to the webcam

    main(video_source)
