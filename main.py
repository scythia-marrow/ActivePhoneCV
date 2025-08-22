import cv2

from ultralytics import YOLO

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
	
	cv2.imwrite(f"out/frame-{fid}.png", frame)
	fid += 1
