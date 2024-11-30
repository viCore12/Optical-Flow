import cv2
import numpy as np
import time

def lucas_kanade_method(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    
    # Start a loop to process each frame
    while True:
        start_time = time.time()  # Record the start time
        
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (360, 288))
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )
        
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
        
        # Add the mask to the frame
        img = cv2.add(frame, mask)
        
        # Calculate FPS
        fps = int(1 / (time.time() - start_time))
        
        # Get the frame size
        height, width = frame.shape[:2]
        
        # Display FPS and Frame size on the frame
        cv2.putText(img, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Size: {width}x{height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the processed frame
        cv2.imshow("frame", img)
        
        # Handle key press events
        k = cv2.waitKey(25) & 0xFF
        if k == 27:  # Escape key
            break
        if k == ord("c"):  # 'c' key to clear the mask
            mask = np.zeros_like(old_frame)
        
        # Update previous frame and points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    cap.release()
    cv2.destroyAllWindows()
