from ultralytics import YOLO

model = YOLO('yolov8x')

# Save the predictions in the current working directory
res = model.track('input/video.mov', save=True, save_dir='.')

# print(res)
# print("boxes:")
# for box in res[0].boxes:
#     print(box)
