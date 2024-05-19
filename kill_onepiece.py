import os
import json
import time

# JSON 파일과 이미지 파일의 경로 설정
json_folder = './data/json'
image_folder = './data/image'

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
    
    # 상의 또는 하의 폴리곤 좌표가 없는 경우 JSON 파일과 대응하는 이미지 파일 삭제
    if not any(upper_coords) and not any(lower_coords):
        image_id = data['이미지 정보']['이미지 식별자']
        image_filename = f"{image_id}.jpg"
        image_path = os.path.join(image_folder, image_filename)
        
        # JSON 파일 삭제
        try:
            os.remove(json_path)
        except PermissionError:
            print(f"Failed to delete {json_path}, retrying in 1 second...")
            time.sleep(1)
            os.remove(json_path)
        
        # 이미지 파일 삭제
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"Deleted: {json_path} and {image_path}")
            except PermissionError:
                print(f"Failed to delete {image_path}, retrying in 1 second...")
                time.sleep(1)
                os.remove(image_path)
        else:
            print(f"Deleted: {json_path}")
