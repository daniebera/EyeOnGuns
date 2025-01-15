import yaml
import shutil
import random
from collections import Counter, defaultdict
import os



"""
Create directories for train, val, and test splits with images and labels subdirectories.
"""
def create_dirs(base_path):
    for dir_name in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_path, dir_name, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, dir_name, 'labels'), exist_ok=True)

"""
Copy files from source to destination.
"""
def copy_files(src, dst, files):
    for f in files:
        src_file = os.path.join(src, f)
        dst_file = os.path.join(dst, f)
        shutil.copy(src_file, dst_file)

"""
Split dataset into train, validation, and test sets based on the specified ratios and isolated features.
"""
def split_dataset(base_path, split_ratios=None, key_feature=None):
    if split_ratios is not None:
        assert round(sum(split_ratios), 10) == 1, "The split ratios must sum to 1."
    create_dirs(base_path)

    categories = ['Handgun', 'Machine_Gun', 'No_Gun']
    all_data = []

    for category in categories:
        category_path = os.path.join(base_path, category)
        subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]

        for subfolder in subfolders:
            info = subfolder.split('_')
            camera = info[1]
            place = info[2]
            subject = info[3]
            brightness = info[4]

            frames_path = os.path.join(category_path, subfolder, 'frames')
            if not os.path.exists(frames_path):
                print(f"Frames path {frames_path} does not exist. Skipping...")
                continue

            images = [f for f in os.listdir(frames_path) if f.endswith('.jpg')]
            labels = [f for f in os.listdir(frames_path) if f.endswith('.txt')]

            all_data.append({
                'category': category,
                'folder': subfolder,
                'frames': frames_path,
                'images': images,
                'labels': labels,
                'subject': subject,
                'brightness': brightness,
                'camera': camera,
                'place': place
            })

    random.shuffle(all_data)

    """
    Split train and validation data based on the specified ratios.
    """
    def split_data_by_feature(data, split_ratios=None, key_feature=None):
        feature_groups = defaultdict(list)
        for item in data:
            # Choose the key feature for splitting
            # key = (item['place'])
            key = (item['camera'])
            # key = (item['category'], item['place'], item['subject'])
            # key = (item['category'], item['place'], item['subject'], item['brightness'], item['camera'])
            # key = (item['camera'])
            # key = ()
            feature_groups[key].append(item)
        for key, items in feature_groups.items():
            num_images = sum(len(item['images']) for item in items)
            num_labels = sum(len(item['labels']) for item in items)
            print(f"Key: {key}, Number of images: {num_images}, Number of labels: {num_labels}")

        train_data, val_data, test_data = [], [], []

        if split_ratios is not None:
            for key, items in feature_groups.items():
                print(f"Splitting data for key: {key}, with {len(items)} items")
                total_items = len(items) # Total number of videos for the key
                train_size = int(total_items * split_ratios[0])
                val_size = int(total_items * split_ratios[1])

                train_data.extend(items[:train_size])
                val_data.extend(items[train_size:train_size + val_size])
                test_data.extend(items[train_size + val_size:])
        else:
            if key_feature == 'place':
                raise NotImplementedError("Splitting by place is not implemented yet.")
            elif key_feature == 'camera':
                for key, items in feature_groups.items():
                    print(f"Splitting data for key: {key}, with {len(items)} items")
                    if key == 'C2':
                        train_data.extend(items)
                    elif key == 'C1':
                        total_items = len(items) # Total number of videos for the key
                        val_size = int(total_items * 0.5)
                        val_data.extend(items[:val_size])
                        test_data.extend(items[val_size:])

        return train_data, val_data, test_data

    train_data, val_data, test_data = split_data_by_feature(all_data, split_ratios=split_ratios, key_feature=key_feature)

    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

    for split, data in splits.items():
        split_images_path = os.path.join(base_path, split, 'images')
        split_labels_path = os.path.join(base_path, split, 'labels')
        os.makedirs(split_images_path, exist_ok=True)
        os.makedirs(split_labels_path, exist_ok=True)

        for item in data:
            category = item['category']
            folder_name = item['folder']
            frames_path = item['frames']
            images = item['images']
            labels = item['labels']

            print(frames_path)

            for img in images:
                frame_number = img.split('.')[0].split('_')[-1]
                new_image_name = f"{category}_{folder_name}_frame_{frame_number}.jpg"
                src_img = os.path.join(frames_path, img)
                dst_img = os.path.join(split_images_path, new_image_name)
                shutil.copy(src_img, dst_img)

            for lbl in labels:
                frame_number = lbl.split('.')[0].split('_')[-1]
                new_label_name = f"{category}_{folder_name}_frame_{frame_number}.txt"
                src_lbl = os.path.join(frames_path, lbl)
                dst_lbl = os.path.join(split_labels_path, new_label_name)
                shutil.copy(src_lbl, dst_lbl)

    print("Data split complete.")
    return splits

"""
Check if frames are assigned to the same split.
"""
def check_frames_in_same_split(splits):
    folder_to_split = {}

    for split, split_data in splits.items():
        for item in split_data:
            folder = item['folder']
            if folder in folder_to_split:
                print(f"Error: Folder {folder} is present in both {folder_to_split[folder]} and {split}")
            else:
                folder_to_split[folder] = split

    print("Verification complete - every folder is assigned to one split.")
    # for folder, split in folder_to_split.items():
    #    print(f"Folder {folder} is correctly assigned to split {split}")

"""
Check the balance of data based on the specified feature.
"""
def check_balance(splits, feature):
    total_counts = Counter()
    split_counts = {'train': Counter(), 'val': Counter(), 'test': Counter()}

    for split, data in splits.items():
        for item in data:
            key = item[feature]
            total_counts[key] += len(item['images'])
            split_counts[split][key] += len(item['images'])

    print(f"\nBalancing based on {feature}:")
    total = sum(total_counts.values())
    print(
        f"{'Feature':<20} {'Total':<10} {'Train':<10} {'Val':<10} {'Test':<10} {'Train %':<10} {'Val %':<10} {'Test %':<10}")
    print("-" * 95)
    for key in sorted(total_counts.keys()):
        train_count = split_counts['train'][key]
        val_count = split_counts['val'][key]
        test_count = split_counts['test'][key]
        train_percent = (train_count / total_counts[key]) * 100 if total_counts[key] > 0 else 0
        val_percent = (val_count / total_counts[key]) * 100 if total_counts[key] > 0 else 0
        test_percent = (test_count / total_counts[key]) * 100 if total_counts[key] > 0 else 0
        print(
            f"{key:<20} {total_counts[key]:<10} {train_count:<10} {val_count:<10} {test_count:<10} {train_percent:<10.2f} {val_percent:<10.2f} {test_percent:<10.2f}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Split dataset into train, validation, and test sets.')
    parser.add_argument('--base_path', type=str, default='../Gun_Action_Recognition_Dataset', help='Base path for the dataset')
    parser.add_argument('--split_ratios', type=float, nargs=3, default=None, help='Split ratios for train, val, and test sets')
    parser.add_argument('--key_feature', type=str, default='camera', help='Feature to split data on')

    args = parser.parse_args()

    splits = split_dataset(args.base_path, args.split_ratios, args.key_feature)

    check_frames_in_same_split(splits)
    check_balance(splits, 'category')
    check_balance(splits, 'place')
    check_balance(splits, 'subject')
    check_balance(splits, 'brightness')
    check_balance(splits, 'camera')
    print("Dataset split and verification complete.")
    # os.makedirs(os.path.dirname(f'data/{args.key_feature}_splits.yaml'), exist_ok=True)
    # with open(f'data/{args.key_feature}_splits.yaml', 'w') as file:
    #     documents = yaml.dump(splits, file)
    # print("Splits saved to splits.yaml")
