import os
import cv2
import json

# Extract & convert

"""
Extract frames from a video and save them as images.
"""
def extract_frames(video_path, output_folder, no_labels=False):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f'frame_{count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        count += 1
    cap.release()
    return count

"""
Convert bounding box format from COCO to YOLO.
"""
def convert_bbox_format(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    x_center = (x_min + width / 2) / img_width
    y_center = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height

"""
Convert COCO annotations to YOLO format and save them.
"""
def convert_labels_json(json_path, frame_folder, class_mapping):
    with open(json_path, 'r') as f:
        data = json.load(f)

    categories = {cat['id']: class_mapping[cat['name']] for cat in data['categories']}
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']

    for ann in annotations:
        image_info = images[ann['image_id']]
        frame_index = image_info['id']  # Assuming image_id corresponds to frame index
        frame_file = os.path.join(frame_folder, f'frame_{frame_index:04d}.jpg')
        if not os.path.exists(frame_file):
            continue

        img_width = image_info['width']
        img_height = image_info['height']
        bbox = ann['bbox']
        class_id = categories[ann['category_id']]  # YOLO class IDs start from 0

        x_center, y_center, width, height = convert_bbox_format(bbox, img_width, img_height)

        yolo_label = f"{class_id} {x_center} {y_center} {width} {height}\n"

        output_label_file = frame_file.replace('.jpg', '.txt')
        with open(output_label_file, 'w') as label_file:
            label_file.write(yolo_label)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract frames and convert labels for the dataset.')
    parser.add_argument('--base_path', type=str, default='../Gun_Action_Recognition_Dataset', help='Base path for the dataset')
    parser.add_argument('--categories', type=str, nargs='+', default=['Handgun', 'Machine_Gun', 'No_Gun'], help='List of categories')
    parser.add_argument('--class_ids', type=json.loads, default='{"Handgun": 0, "Machine_Gun": 1, "No_Gun": 2}', help='Class IDs mapping')

    args = parser.parse_args()

    base_path = args.base_path
    categories = args.categories
    class_ids = args.class_ids

    for category in categories:
        category_path = os.path.join(base_path, category)
        for subdir in os.listdir(category_path):
            subdir_path = os.path.join(category_path, subdir)
            if os.path.isdir(subdir_path):  # Check if the subdir_path is a directory
                video_path = os.path.join(subdir_path, "video.mp4")
                frame_folder = os.path.join(subdir_path, 'frames')
                os.makedirs(frame_folder, exist_ok=True)

                print(f"Processing video: {video_path}")
                print(f"Frame folder: {frame_folder}")

                # Extract frames
                extract_frames(video_path, frame_folder, no_labels=(category == 'No_Gun'))

                # Convert annotations only if not 'No_Gun'
                if category != 'No_Gun':
                    json_path = os.path.join(subdir_path, 'label.json')
                    convert_labels_json(json_path, frame_folder, class_ids)
            else:
                print(f"Skipping non-directory item: {subdir_path}")