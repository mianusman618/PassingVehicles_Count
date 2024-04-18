from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

model = YOLO("./yolov8n.pt")
cap = cv2.VideoCapture("./video.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

#line_points = [(500, 400), (900, 400)]  # line or region points
line_points = [(500, 400), (900, 400), (900, 360), (500, 360)]
classes_to_count = [2]  # person and car classes for count

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.mp4",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True,
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True,
                 line_thickness=2,
                 view_in_counts=False)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=False,
                         classes=classes_to_count)
    #print(tracks)
    im0 = counter.start_counting(im0, tracks)
    print(counter.out_counts)
    counter.in_counts=counter.in_counts+counter.out_counts
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()