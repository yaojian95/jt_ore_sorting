import cv2
import numpy as np
import argparse
import os

def stack_tif(data_path, height_factor):
    '''
    Walking through all the subfolders and stack all the tif in that subfolder to images 
    '''

    for root, dirs, files in os.walk(data_path):
        image_data = []
        if len(files) > 1:
            # print(files)
            files.sort()
            print(files)

            for file in files:
                if file.lower().endswith('.tif'):
                #     print(root.split('/')[-1], file)
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    image_data.append(img)
                    
            if len(image_data) != 0:
                stacked_image = np.concatenate(image_data, axis=0)

                height, width = stacked_image.shape[:2]
                new_height = height // height_factor
                new_size = (width, new_height)
                resized_img = cv2.resize(stacked_image, new_size, interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(os.path.join(root, '%s.png'%(root.split('/')[-1])), resized_img)    

def stack_images(folder_path, start_idx=None, end_idx=None, prefix=3):
    '''
    stack cettain tifs into images 
    '''
    
    files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    files.sort(key=lambda x: int(x.split('.')[0][prefix:]))
    image_data = []
    start_idx = 1 if start_idx is None else start_idx
    end_idx = len(files) if end_idx is None else end_idx

    for file in files[start_idx - 1:end_idx]:
        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        image_data.append(img)

    stacked_image = np.concatenate(image_data, axis=0)
    return stacked_image

def main():
    parser = argparse.ArgumentParser(description='Stack TIFF images vertically')
    parser.add_argument('date', help='Experiment date in YYYYMMDD format')
    parser.add_argument('para', help='Experiment parameters')
    parser.add_argument('start_index', type=int, help='Start image index')
    parser.add_argument('end_index', type=int, help='End image index')
    
    args = parser.parse_args()

    base_dir = os.path.join("E:/luoyihan", args.date, args.para)
    input_folder = os.path.join(base_dir, "Images")
    output_path = os.path.join(base_dir, f"{args.para}.png")

    stacked_image = stack_images(input_folder, args.start_index, args.end_index)
    cv2.imwrite(output_path, stacked_image)
    # print(f"Successfully saved stacked image to: {output_path}")

if __name__ == "__main__":
    main()