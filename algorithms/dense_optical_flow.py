import cv2
import numpy as np
import time

def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # read the video
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, old_frame = cap.read()

    # Resize the first frame to 360x288
    old_frame = cv2.resize(old_frame, (360, 288))
    
    # Create HSV image and make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        start_time = time.time()  # Record the start time

        # Read the next frame
        ret, new_frame = cap.read()
        
        if not ret:
            break

        # Resize the new frame to 360x288
        new_frame = cv2.resize(new_frame, (360, 288))

        frame_copy = new_frame
        
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Convert the flow to Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the HSV image to BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Calculate FPS
        fps = int(1 / (time.time() - start_time))

        # Get the frame size (width and height)
        height, width = frame_copy.shape[:2]

        # Display FPS and Frame size on the frame
        cv2.putText(frame_copy, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_copy, f"Size: {width}x{height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame and the optical flow
        cv2.imshow("frame", frame_copy)
        cv2.imshow("optical flow", bgr)

        # Handle key press events
        k = cv2.waitKey(25) & 0xFF
        if k == 27:  # Escape key
            break
        if k == ord("c"):  # 'c' key to clear the mask or reset
            hsv[..., 2] = 0  # Reset the saturation to clear flow visualization

        # Update previous frame
        old_frame = new_frame

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
