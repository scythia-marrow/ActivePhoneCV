import cv2

from ultralytics import YOLO

def bbox_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
        box1 (tuple): (x_min, y_min, x_max, y_max) for the first box
        box2 (tuple): (x_min, y_min, x_max, y_max) for the second box
    
    Returns:
        float: IoU value between 0 and 1
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

    # Compute the IoU
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0  # avoid division by zero
    return inter_area / union_area


model = YOLO("yoloe-11l-seg.pt")
names = ["person", "cell phone", "red phone", "green phone", "blue phone", "black phone", "white phone", "person hand"]
model.set_classes(names, model.get_text_pe(names))


capture = cv2.VideoCapture("people_with_phones.mp4")

fid = 0
while capture.isOpened():
	success, frame = capture.read()
	if not success: break
	results = model.predict(frame)

	num_active = 0
	active = []
	for bbox in results[0].boxes:
		np = bbox.xyxy.cpu().flatten(0)
		cut = frame[int(np[1]):int(np[3]),int(np[0]):int(np[2])]
		res = model.predict(cut)
		phone_box = []
		for x in res:
			if len(x.boxes) == 0: continue
			for b in x.boxes:
				if "phone" not in x.names[int(b.cls)]: continue
				phone_box.append(b)
		# TODO: most central phone or something?
		if len(phone_box) > 0:
			active.append((np, phone_box[0].xyxy.cpu().flatten(0)))

	for np, a in active:
		print(np, a)
		box = [int(np[0] + a[0]), int(np[1] + a[1]), int(np[0] + a[2]), int(np[1] + a[3])]
		cv2.rectangle(frame, (box[0],box[1]), (box[2],box[3]), color=(0,0,255), thickness=2)
	
	cv2.imshow("Active phones", cv2.resize(frame, (960,480)))
	cv2.waitKey()
