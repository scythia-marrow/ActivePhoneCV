import cv2

from ultralytics import YOLO

model = YOLO("yoloe-11l-seg.pt")
names = ["person", "cell phone", "red phone", "green phone", "blue phone", "black phone", "white phone", "person hand"]
model.set_classes(names, model.get_text_pe(names))

def bbox_diff(box1, box2):
    """
    Calculate the left-norm intersection (IoL) of two bounding boxes.
    
    Parameters:
        box1 (tuple): (x_min, y_min, x_max, y_max) for the first box
        box2 (tuple): (x_min, y_min, x_max, y_max) for the second box
    
    Returns:
        float: IoL value between 0 and 1
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Determine the coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Compute the area of intersection
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    inter_area = inter_width * inter_height

    # Compute the area of both bounding boxes
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute the IoL
    if area1 == 0:
        return 0.0  # avoid division by zero
    return inter_area / area1

capture = cv2.VideoCapture("people_with_phones.mp4")


fid = 0
while capture.isOpened():
	success, frame = capture.read()
	if not success: break
	results = model.predict(frame)

	num_active = 0
	person = []
	phone = []
	active = []
	for bbox in results[0].boxes:
		if int(bbox.cls) == 0: person.append(bbox)
		if int(bbox.cls) > 0 and int(bbox.cls) < 7: phone.append(bbox)
	
	for pbox in person:
		for fbox in phone:
			pflat = pbox.xyxy.cpu().flatten(0)
			fflat = fbox.xyxy.cpu().flatten(0)
			iol = bbox_diff(fflat, pflat)
			print(iol)
			if iol > 0.8: active.append(fflat)

	for a in active:
		cv2.rectangle(frame, (int(a[0]),int(a[1])), (int(a[2]),int(a[3])), color=(0,0,255), thickness=2)
	
	cv2.imwrite(f"out/frame-{fid}.png", frame)
	fid += 1
