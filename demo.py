import numpy as np
import torch
import cv2
import sys
import time
import threading
from collections import deque

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazepalm import BlazePalm
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark

from visualization import (
    draw_detections,
    draw_landmarks,
    draw_roi,
    HAND_CONNECTIONS,
    FACE_CONNECTIONS,
)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")
torch.set_grad_enabled(False)

back_detector = True
face_detector = BlazeFace(back_model=back_detector).to(device)
if back_detector:
    face_detector.load_weights("blazefaceback.pth")
    face_detector.load_anchors("anchors_face_back.npy")
else:
    face_detector.load_weights("blazeface.pth")
    face_detector.load_anchors("anchors_face.npy")

palm_detector = BlazePalm().to(device)
palm_detector.load_weights("blazepalm.pth")
palm_detector.load_anchors("anchors_palm.npy")
palm_detector.min_score_thresh = 0.75

hand_regressor = BlazeHandLandmark().to(device)
hand_regressor.load_weights("blazehand_landmark.pth")

face_regressor = BlazeFaceLandmark().to(device)
face_regressor.load_weights("blazeface_landmark.pth")

detect_face = False
detect_hand = True


class ThreadedCamera:
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Set camera properties for higher FPS
        self.capture.set(cv2.CAP_PROP_FPS, 60)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.frame_buffer = deque(maxlen=2)  # Keep only latest 2 frames
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.daemon = True  # Dies when main thread dies
        self.thread.start()

    def update(self):
        while self.running:
            try:
                ret, frame = self.capture.read()
                if ret:
                    if len(self.frame_buffer) == self.frame_buffer.maxlen:
                        self.frame_buffer.popleft()  # Remove oldest frame
                    self.frame_buffer.append(frame)
            except Exception:
                break

    def read(self):
        if len(self.frame_buffer) > 0:
            return True, self.frame_buffer[-1]  # Return latest frame
        return False, None

    def stop(self):
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)  # Don't wait forever
        self.capture.release()


WINDOW = "test"
cv2.namedWindow(WINDOW)
if len(sys.argv) > 1:
    capture = ThreadedCamera(sys.argv[1])
    mirror_img = False
else:
    capture = ThreadedCamera(0)
    mirror_img = True

capture.start()
time.sleep(0.1)  # Give camera time to start

hasFrame, frame = capture.read()
if frame is not None:
    hasFrame = True
    frame_ct = 0
else:
    hasFrame = False

# FPS tracking variables
fps = 0
prev_time = time.time() - 1e-6  # Start with a small offset to avoid division by zero

try:
    while hasFrame:
        frame_ct += 1

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        if mirror_img:
            frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
        else:
            frame = np.ascontiguousarray(frame[:, :, ::-1])

        img1, img2, scale, pad = resize_pad(frame)

        # Face detection and landmarks
        if detect_face:
            if back_detector:
                normalized_face_detections = face_detector.predict_on_image(img1)
            else:
                normalized_face_detections = face_detector.predict_on_image(img2)
            face_detections = denormalize_detections(
                normalized_face_detections, scale, pad
            )

            xc, yc, scale_face, theta = face_detector.detection2roi(
                face_detections,
            )
            img_face, affine, box = face_regressor.extract_roi(
                frame, xc, yc, theta, scale_face
            )
            flags, normalized_landmarks = face_regressor(img_face.to(device))
            landmarks = face_regressor.denormalize_landmarks(
                normalized_landmarks, affine
            )

            for i in range(len(flags)):
                landmark, flag = landmarks[i], flags[i]
                if flag > 0.5:
                    draw_landmarks(frame, landmark[:, :2], FACE_CONNECTIONS, size=1)

            draw_roi(frame, box)
            draw_detections(frame, face_detections)

        # Hand detection and landmarks
        if detect_hand:
            normalized_palm_detections = palm_detector.predict_on_image(img1)
            palm_detections = denormalize_detections(
                normalized_palm_detections, scale, pad
            )

            xc, yc, scale_hand, theta = palm_detector.detection2roi(palm_detections)
            img_hand, affine2, box2 = hand_regressor.extract_roi(
                frame, xc, yc, theta, scale_hand
            )
            flags2, handed2, normalized_landmarks2 = hand_regressor(img_hand.to(device))
            landmarks2 = hand_regressor.denormalize_landmarks(
                normalized_landmarks2,
                affine2,
            )

            for i in range(len(flags2)):
                landmark, flag = landmarks2[i], flags2[i]
                if flag > 0.5:
                    draw_landmarks(frame, landmark[:, :2], HAND_CONNECTIONS, size=2)

            draw_roi(frame, box2)
            draw_detections(frame, palm_detections)

        # Draw FPS in top-left corner
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow(WINDOW, frame[:, :, ::-1])

        hasFrame, frame = capture.read()
        key = cv2.waitKey(1)
        if key == 27:
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")
finally:
    capture.stop()
    cv2.destroyAllWindows()
