import os
import json
import cv2
import numpy as np

def crop_and_save(image_path, coords, save_path, size=(400, 400)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    # Validate coordinates to be within the image bounds
    height, width = image.shape[:2]
    valid_coords = [(x, y) for x, y in coords if 0 <= x < width and 0 <= y < height]
    
    if not valid_coords:
        print(f"No valid coordinates for {image_path}")
        return

    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = np.array(valid_coords, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)

    # Extract region from mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    cropped_image = masked_image[y:y+h, x:x+w]

    if cropped_image.size == 0:
        print(f"Cropped image is empty for {image_path}")
        return

    # Create transparent background
    bgra = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
    alpha = mask[y:y+h, x:x+w]
    bgra[:, :, 3] = alpha

    # Resize to desired size
    resized_image = cv2.resize(bgra, size, interpolation=cv2.INTER_AREA)

    # Save the cropped and resized image
    cv2.imwrite(save_path, resized_image)

# JSON 파일과 이미지 파일의 경로 설정
json_folder = './data/json'
image_folder = './data/image'
top_crop_image_folder = './data/top/crop_image'
top_json_folder = './data/top/json'
bottom_crop_image_folder = './data/bottom/crop_image'
bottom_json_folder = './data/bottom/json'

os.makedirs(top_crop_image_folder, exist_ok=True)
os.makedirs(top_json_folder, exist_ok=True)
os.makedirs(bottom_crop_image_folder, exist_ok=True)
os.makedirs(bottom_json_folder, exist_ok=True)

# JSON 파일들을 가져오기
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)
    
    # JSON 파일 열기
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    # 폴리곤 좌표 가져오기
    polygon_coords = data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']
    
    # 상의와 하의의 폴리곤 좌표가 있는지 확인
    upper_coords = polygon_coords.get('상의', [{}])
    lower_coords = polygon_coords.get('하의', [{}])
    
    image_filename = '%d.jpg' % data['이미지 정보']['이미지 식별자']
    image_path = os.path.join(image_folder, image_filename)
    
    if any(upper_coords):
        upper_points = [(coord[f"X좌표{i}"], coord[f"Y좌표{i}"]) for coord in upper_coords for i in range(1, 49) if f"X좌표{i}" in coord and f"Y좌표{i}" in coord]
        if upper_points:
            upper_save_path = os.path.join(top_crop_image_folder, image_filename)
            crop_and_save(image_path, upper_points, upper_save_path)
            
            upper_json_path = os.path.join(top_json_folder, json_file)
            with open(upper_json_path, 'w', encoding='utf-8') as upper_file:
                json.dump(data, upper_file, ensure_ascii=False, indent=4)
        
    if any(lower_coords):
        lower_points = [(coord[f"X좌표{i}"], coord[f"Y좌표{i}"]) for coord in lower_coords for i in range(1, 49) if f"X좌표{i}" in coord and f"Y좌표{i}" in coord]
        if lower_points:
            lower_save_path = os.path.join(bottom_crop_image_folder, image_filename)
            crop_and_save(image_path, lower_points, lower_save_path)
            
            lower_json_path = os.path.join(bottom_json_folder, json_file)
            with open(lower_json_path, 'w', encoding='utf-8') as lower_file:
                json.dump(data, lower_file, ensure_ascii=False, indent=4)
