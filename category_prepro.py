import os
import json
import cv2
import numpy as np

# 카테고리 정의
top_categories = ["탑", "블라우스", "니트웨어", "셔츠", "티셔츠", "브라탑", "후드"]
bottom_categories = ["스커트", "청바지", "팬츠", "레깅스", "조거팬츠"]

def crop_and_save(image_path, coords, save_path, size=(400, 400)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return False

    # Validate coordinates to be within the image bounds
    height, width = image.shape[:2]
    valid_coords = [(x, y) for x, y in coords if 0 <= x < width and 0 <= y < height]
    
    if not valid_coords:
        print(f"No valid coordinates for {image_path}")
        return False

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
        return False

    # Create transparent background
    bgra = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
    alpha = mask[y:y+h, x:x+w]
    bgra[:, :, 3] = alpha

    # Resize to desired size
    resized_image = cv2.resize(bgra, size, interpolation=cv2.INTER_AREA)

    # Save the cropped and resized image
    cv2.imwrite(save_path, resized_image)
    return True

def get_category_counts(categories):
    return {category: 0 for category in categories}

# 경로 설정
json_folder = './data/json'
image_folder = './data/image'
base_top_crop_image_folder = './data/top/crop_image'
base_top_json_folder = './data/top/json'
base_bottom_crop_image_folder = './data/bottom/crop_image'
base_bottom_json_folder = './data/bottom/json'

# 카테고리 폴더 생성
for category in top_categories:
    os.makedirs(os.path.join(base_top_crop_image_folder, category), exist_ok=True)
    os.makedirs(os.path.join(base_top_json_folder, category), exist_ok=True)
for category in bottom_categories:
    os.makedirs(os.path.join(base_bottom_crop_image_folder, category), exist_ok=True)
    os.makedirs(os.path.join(base_bottom_json_folder, category), exist_ok=True)

# JSON 파일 목록 가져오기
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

# 카테고리별 이미지 및 JSON 파일 개수 추적
top_counts = get_category_counts(top_categories)
bottom_counts = get_category_counts(bottom_categories)

for json_file in json_files:
    json_path = os.path.join(json_folder, json_file)
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    polygon_coords = data['데이터셋 정보']['데이터셋 상세설명']['폴리곤좌표']
    labeling = data['데이터셋 정보']['데이터셋 상세설명']['라벨링']
    
    image_filename = '%d.jpg' % data['이미지 정보']['이미지 식별자']
    image_path = os.path.join(image_folder, image_filename)
    
    upper_coords = polygon_coords.get('상의', [{}])
    lower_coords = polygon_coords.get('하의', [{}])
    
    if any(upper_coords):
        upper_points = [(coord.get(f"X좌표{i}"), coord.get(f"Y좌표{i}")) for coord in upper_coords for i in range(1, 49) if f"X좌표{i}" in coord and f"Y좌표{i}" in coord]
        upper_category = labeling.get('상의', [{}])[0].get('카테고리', None)
        
        if upper_points and upper_category in top_categories:
            upper_save_path = os.path.join(base_top_crop_image_folder, upper_category, image_filename)
            if crop_and_save(image_path, upper_points, upper_save_path):
                upper_json_path = os.path.join(base_top_json_folder, upper_category, json_file)
                with open(upper_json_path, 'w', encoding='utf-8') as upper_file:
                    json.dump(data, upper_file, ensure_ascii=False, indent=4)
                top_counts[upper_category] += 1
        
    if any(lower_coords):
        lower_points = [(coord.get(f"X좌표{i}"), coord.get(f"Y좌표{i}")) for coord in lower_coords for i in range(1, 49) if f"X좌표{i}" in coord and f"Y좌표{i}" in coord]
        lower_category = labeling.get('하의', [{}])[0].get('카테고리', None)
        
        if lower_points and lower_category in bottom_categories:
            lower_save_path = os.path.join(base_bottom_crop_image_folder, lower_category, image_filename)
            if crop_and_save(image_path, lower_points, lower_save_path):
                lower_json_path = os.path.join(base_bottom_json_folder, lower_category, json_file)
                with open(lower_json_path, 'w', encoding='utf-8') as lower_file:
                    json.dump(data, lower_file, ensure_ascii=False, indent=4)
                bottom_counts[lower_category] += 1

# 카테고리별 분포 출력
print("Top Categories Distribution:")
for category, count in top_counts.items():
    print(f"{category}: {count}")

print("\nBottom Categories Distribution:")
for category, count in bottom_counts.items():
    print(f"{category}: {count}")
