import json
import os
import shutil
from typing import List, Set
from PIL import Image
import time # Thêm thư viện time
# ĐỔI TÊN HÀM CŨ (Hàm 4) THÀNH HÀM NÀY (Hàm 1 mới)
def sync_json_with_source_images(
    json_path: str,
    image_dir: str,
    output_json_path: str,
    image_ext_in_dir: str = ".png",
    image_ext_in_json: str = ".jpg"
):
    """
    Hàm 1 (Mới): Lọc file JSON (captions/instances) GỐC dựa trên các
    file ảnh thực tế có trong thư mục ảnh GỐC.
    """
    print(f"--- Bắt đầu Hàm 1: Đồng bộ file '{os.path.basename(json_path)}' ---")
    print(f"    Dựa trên ảnh thực tế có trong: {image_dir}")

    # 1. Lấy danh sách tên file "ground truth" từ thư mục ảnh gốc
    try:
        actual_filenames_with_ext = os.listdir(image_dir)
    except FileNotFoundError:
        print(f"  LỖI: Không tìm thấy thư mục ảnh nguồn: {image_dir}")
        return False # Trả về False nếu lỗi

    # 2. Chuyển đổi tên file này sang định dạng trong JSON
    valid_filenames_in_json_format: Set[str] = set()
    for fname in actual_filenames_with_ext:
        if fname.endswith(image_ext_in_dir):
            base_name, _ = os.path.splitext(fname)
            json_filename = base_name + image_ext_in_json
            valid_filenames_in_json_format.add(json_filename)
    
    if not valid_filenames_in_json_format:
        print(f"  CẢNH BÁO: Không tìm thấy ảnh nào với đuôi '{image_ext_in_dir}' trong {image_dir}.")
        return False

    print(f"    Tìm thấy {len(valid_filenames_in_json_format)} ảnh hợp lệ trong thư mục nguồn.")

    # 3. Tải file JSON (file GỐC)
    print(f"    Đang tải file JSON gốc: {json_path} (Việc này có thể mất thời gian)")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"  LỖI: Không tìm thấy file JSON gốc: {json_path}")
        return False
    print(f"    Đã tải xong. Bắt đầu lọc...")

    # 4. Lọc 'images'
    original_image_count = len(data['images'])
    filtered_images: List[dict] = []
    for img in data['images']:
        if img['file_name'] in valid_filenames_in_json_format:
            filtered_images.append(img)
    
    print(f"    Đã lọc 'images'. Giữ lại {len(filtered_images)} / {original_image_count} ảnh.")

    # 5. Lấy ID của các ảnh được giữ lại
    valid_image_ids: Set[int] = {img['id'] for img in filtered_images}
    
    # 6. Lọc 'annotations'
    original_annotation_count = len(data['annotations'])
    filtered_annotations: List[dict] = []
    for ann in data['annotations']:
        if ann['image_id'] in valid_image_ids:
            filtered_annotations.append(ann)

    print(f"    Đã lọc 'annotations'. Giữ lại {len(filtered_annotations)} / {original_annotation_count} chú thích.")

    # 7. Xây dựng và lưu file JSON (file trung gian)
    output_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': data.get('categories', []) 
    }

    print(f"    Đang lưu file JSON đã đồng bộ vào: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f)
        
    print(f"--- Hoàn thành Hàm 1 cho file '{os.path.basename(json_path)}' ---")
    return True

def filter_captions_by_keywords(
    captions_json_path: str,
    keywords: List[str],
    output_json_path: str
) -> Set[int]:
    """
    Hàm 1: Lọc file JSON captions dựa trên danh sách từ khóa.
    
    Tìm tất cả các caption có chứa bất kỳ từ khóa nào (không phân biệt chữ hoa/thường).
    Tạo ra một file JSON mới chỉ chứa các 'images' và 'annotations' (caption)
    khớp.
    """
    print(f"--- Bắt đầu Hàm 1: Lọc file captions '{captions_json_path}' ---")
    
    # Chuẩn hóa từ khóa về chữ thường
    lower_keywords = [kw.lower() for kw in keywords]
    
    print(f"Đang tải file JSON... (việc này có thể mất một lúc)")
    with open(captions_json_path, 'r', encoding='utf-8') as f:
        captions_data = json.load(f)
    
    print("Đã tải xong. Bắt đầu lọc...")
    
    matching_image_ids: Set[int] = set()
    filtered_caption_annotations: List[dict] = []
    
    # 1. Tìm tất cả các caption annotation khớp
    total_captions = len(captions_data['annotations'])
    for i, ann in enumerate(captions_data['annotations']):
        if (i + 1) % 50000 == 0:
            print(f"  Đã kiểm tra {i+1}/{total_captions} captions...")
            
        caption_text = ann['caption'].lower()
        # Kiểm tra xem có từ khóa nào nằm trong caption không
        if any(kw in caption_text for kw in lower_keywords):
            filtered_caption_annotations.append(ann)
            matching_image_ids.add(ann['image_id'])
            
    print(f"Tìm thấy {len(filtered_caption_annotations)} captions khớp với {len(matching_image_ids)} ảnh.")

    # 2. Lọc danh sách 'images' dựa trên các ID đã tìm thấy
    filtered_images: List[dict] = []
    for img in captions_data['images']:
        if img['id'] in matching_image_ids:
            filtered_images.append(img)
            
    print(f"Đã lọc danh sách 'images'. Giữ lại {len(filtered_images)} ảnh.")

    # 3. Tạo file JSON output mới
    output_data = {
        'info': captions_data.get('info', {}),
        'licenses': captions_data.get('licenses', []),
        'images': filtered_images,
        'annotations': filtered_caption_annotations
    }
    
    # 4. Lưu file JSON mới
    print(f"Đang lưu file JSON đã lọc vào: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f)
        
    print(f"--- Hoàn thành Hàm 1 ---")
    return matching_image_ids

def filter_instances_by_captions(
    instances_json_path: str,
    filtered_captions_json_path: str,
    output_json_path: str
):
    """
    Hàm 2: Lọc file JSON instances dựa trên file captions đã lọc.
    
    Sử dụng danh sách 'images' từ file captions đã lọc để giữ lại
    'images' và 'annotations' (instances) tương ứng trong file instances.
    Tất cả 'categories' sẽ được giữ lại.
    """
    print(f"--- Bắt đầu Hàm 2: Lọc file instances '{instances_json_path}' ---")

    # 1. Lấy danh sách ID ảnh cần giữ từ file captions đã lọc
    print(f"Đang tải file captions đã lọc: {filtered_captions_json_path}")
    with open(filtered_captions_json_path, 'r', encoding='utf-8') as f:
        filtered_captions_data = json.load(f)
    
    image_ids_to_keep: Set[int] = set()
    for img in filtered_captions_data['images']:
        image_ids_to_keep.add(img['id'])
        
    print(f"Cần giữ lại {len(image_ids_to_keep)} ảnh.")

    # 2. Tải file instances (file này rất lớn)
    print(f"Đang tải file instances... (việc này có thể mất một lúc)")
    with open(instances_json_path, 'r', encoding='utf-8') as f:
        instances_data = json.load(f)
    print("Đã tải xong. Bắt đầu lọc...")

    # 3. Lọc danh sách 'images'
    filtered_images: List[dict] = []
    for img in instances_data['images']:
        if img['id'] in image_ids_to_keep:
            filtered_images.append(img)
    print(f"Đã lọc 'images' (giữ lại {len(filtered_images)}).")

    # 4. Lọc danh sách 'annotations' (instance annotations)
    filtered_instance_annotations: List[dict] = []
    total_annotations = len(instances_data['annotations'])
    for i, ann in enumerate(instances_data['annotations']):
        if (i + 1) % 100000 == 0:
            print(f"  Đã kiểm tra {i+1}/{total_annotations} instance annotations...")
        
        if ann['image_id'] in image_ids_to_keep:
            filtered_instance_annotations.append(ann)
    print(f"Đã lọc 'annotations' (giữ lại {len(filtered_instance_annotations)}).")

    # 5. Tạo file JSON output mới
    # Quan trọng: Giữ lại tất cả 'categories' để file dataset hợp lệ
    output_data = {
        'info': instances_data.get('info', {}),
        'licenses': instances_data.get('licenses', []),
        'images': filtered_images,
        'annotations': filtered_instance_annotations,
        'categories': instances_data.get('categories', []) 
    }
    
    # 6. Lưu file JSON mới
    print(f"Đang lưu file instances đã lọc vào: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f)
        
    print(f"--- Hoàn thành Hàm 2 ---")

def copy_filtered_images(
    filtered_json_path: str,
    source_image_dir: str,
    output_image_dir: str,
    target_extension: str = ".png" # Thêm tham số đuôi file
):
    """
    Hàm 3: Sao chép các file ảnh (sketch) dựa trên file JSON đã lọc.
    
    Đọc 'file_name' (ví dụ: ...jpg) từ JSON, đổi đuôi file
    thành 'target_extension' (ví dụ: ...png), sau đó sao chép.
    """
    print(f"--- Bắt đầu Hàm 3: Sao chép ảnh sketch ---")

    # 1. Đảm bảo thư mục đích tồn tại
    os.makedirs(output_image_dir, exist_ok=True)
    print(f"Thư mục đích: {output_image_dir}")

    # 2. Tải file JSON đã lọc
    print(f"Đang tải file JSON: {filtered_json_path}")
    with open(filtered_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 3. Lấy danh sách tên file
    file_names_from_json: List[str] = [img['file_name'] for img in data['images']]
    print(f"Tìm thấy {len(file_names_from_json)} ảnh cần xử lý.")
    
    # 4. Bắt đầu sao chép
    copied_count = 0
    not_found_count = 0
    for i, original_file_name in enumerate(file_names_from_json):
        if (i + 1) % 1000 == 0:
            print(f"  Đã xử lý {i+1}/{len(file_names_from_json)} ảnh...")
            
        # Tách lấy tên file (không có đuôi)
        # Ví dụ: '000000123456.jpg' -> '000000123456'
        base_name, _ = os.path.splitext(original_file_name)
        
        # Tạo tên file mới với đuôi .png
        target_file_name = base_name + target_extension
        
        # Xây dựng đường dẫn
        source_path = os.path.join(source_image_dir, target_file_name)
        dest_path = os.path.join(output_image_dir, target_file_name)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            copied_count += 1
        else:
            print(f"  Cảnh báo: Không tìm thấy file nguồn: {source_path}")
            not_found_count += 1
            
    print(f"--- Hoàn thành Hàm 3 ---")
    print(f"Tổng kết: Đã sao chép {copied_count} file. Không tìm thấy {not_found_count} file.")

def resize_images(
    source_dir: str,
    output_dir: str,
    size: tuple = (256, 256),
    extension: str = ".png"
):
    """
    Hàm 5 (Mới): Resize tất cả ảnh trong thư mục nguồn về kích thước
    mong muốn và lưu vào thư mục đích.
    """
    print(f"--- Bắt đầu Hàm 5: Resize ảnh về {size} ---")
    
    # 1. Đảm bảo thư mục đích tồn tại
    os.makedirs(output_dir, exist_ok=True)
    print(f"    Thư mục output: {output_dir}")

    try:
        filenames = [f for f in os.listdir(source_dir) if f.endswith(extension)]
    except FileNotFoundError:
        print(f"  LỖI: Không tìm thấy thư mục ảnh nguồn: {source_dir}")
        return
        
    print(f"    Tìm thấy {len(filenames)} ảnh '{extension}' để resize.")
    start_time = time.time()
    
    for i, filename in enumerate(filenames):
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(output_dir, filename)
        
        try:
            # Mở ảnh
            with Image.open(source_path) as img:
                # Resize ảnh. 
                # Image.LANCZOS là thuật toán downsampling chất lượng cao
                img_resized = img.resize(size, Image.LANCZOS)
                
                # Lưu ảnh đã resize
                img_resized.save(dest_path)
                
        except Exception as e:
            print(f"  LỖI khi xử lý file {filename}: {e}")

        if (i + 1) % 100 == 0:
            print(f"    Đã resize {i+1} / {len(filenames)} ảnh...")

    end_time = time.time()
    print(f"--- Hoàn thành Hàm 5 ---")
    print(f"    Đã resize {len(filenames)} ảnh trong {end_time - start_time:.2f} giây.")
# -----------------------------------------------------------------
# CÁCH SỬ DỤNG (QUY TRÌNH MỚI)
# -----------------------------------------------------------------
# -----------------------------------------------------------------
# CÁCH SỬ DỤNG (QUY TRÌNH MỚI + RESIZE)
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    # 1. ĐỊNH NGHĨA CÁC THAM SỐ
    
    KEYWORDS_TO_FILTER = ["giraffe", "bird", "elephant", "cow"]
    TARGET_IMAGE_SIZE = (256, 256) # Kích thước 256x256

    # --- ĐƯỜNG DẪN GỐC ---
    ORIGINAL_CAPTIONS_PATH = r"C:\Users\PC\OneDrive\Desktop\scientific research\model\training_dataset\annotations\captions_train2017.json"
    ORIGINAL_INSTANCES_PATH = r"C:\Users\PC\OneDrive\Desktop\scientific research\model\training_dataset\annotations\instances_train2017.json"
    SOURCE_SKETCH_DIR = r"C:\Users\PC\OneDrive\Desktop\scientific research\model\training_dataset\sketch_images" # Thư mục ảnh sketch gốc (.png)

    # --- ĐƯỜNG DẪN OUTPUT (CUỐI CÙNG) ---
    OUTPUT_ROOT_DIR = "training_dataset_filtered"
    FILTERED_ANNOTATIONS_DIR = os.path.join(OUTPUT_ROOT_DIR, "annotations")
    FINAL_CAPTIONS_PATH = os.path.join(FILTERED_ANNOTATIONS_DIR, "captions_train2017.json")
    FINAL_INSTANCES_PATH = os.path.join(FILTERED_ANNOTATIONS_DIR, "instances_train2017.json")
    
    # Thư mục chứa ảnh đã lọc (kích thước gốc)
    OUTPUT_SKETCH_DIR = os.path.join(OUTPUT_ROOT_DIR, r"C:\Users\PC\OneDrive\Desktop\scientific research\model\training_dataset_filtered\val_dataset")
    # Thư mục mới chứa ảnh đã resize 256x256
    OUTPUT_SKETCH_RESIZED_DIR = os.path.join(OUTPUT_ROOT_DIR, f"val_sketch_images_{TARGET_IMAGE_SIZE[0]}px")

    
    # --- ĐƯỜNG DẪN TẠM (TRUNG GIAN) ---
    TEMP_DIR = "annotations_temp"
    SYNCED_CAPTIONS_PATH = os.path.join(TEMP_DIR, "captions_synced.json")
    SYNCED_INSTANCES_PATH = os.path.join(TEMP_DIR, "instances_synced.json")

    # --- CẤU HÌNH ĐUÔI FILE ---
    SKETCH_FILE_EXTENSION = ".png"
    JSON_FILE_EXTENSION = ".jpg"
    
    # Tạo các thư mục cần thiết
    os.makedirs(FILTERED_ANNOTATIONS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    
    # 2. CHẠY QUY TRÌNH LỌC
    
    print("--- BẮT ĐẦU QUY TRÌNH LỌC DỮ LIỆU ---")

    # # --- HÀM 1: Đồng bộ 2 file JSON GỐC với thư mục ảnh sketch GỐC ---
    # if not os.path.exists(SYNCED_CAPTIONS_PATH) or not os.path.exists(SYNCED_INSTANCES_PATH):
    #     sync_json_with_source_images(
    #         json_path=ORIGINAL_CAPTIONS_PATH,
    #         image_dir=SOURCE_SKETCH_DIR,
    #         output_json_path=SYNCED_CAPTIONS_PATH,
    #         image_ext_in_dir=SKETCH_FILE_EXTENSION,
    #         image_ext_in_json=JSON_FILE_EXTENSION
    #     )
    #     sync_json_with_source_images(
    #         json_path=ORIGINAL_INSTANCES_PATH,
    #         image_dir=SOURCE_SKETCH_DIR,
    #         output_json_path=SYNCED_INSTANCES_PATH,
    #         image_ext_in_dir=SKETCH_FILE_EXTENSION,
    #         image_ext_in_json=JSON_FILE_EXTENSION
    #     )
    # else:
    #     print("--- Bỏ qua Hàm 1 (Đã có file tạm) ---")


    # # --- HÀM 2: Lọc file captions (đã đồng bộ) theo TỪ KHÓA ---
    # print("\n--- Bắt đầu Hàm 2: Lọc captions theo từ khóa ---")
    # filter_captions_by_keywords(
    #     captions_json_path=SYNCED_CAPTIONS_PATH,
    #     keywords=KEYWORDS_TO_FILTER,
    #     output_json_path=FINAL_CAPTIONS_PATH
    # )
    
    # # --- HÀM 3: Lọc file instances (đã đồng bộ) theo file captions (đã lọc) ---
    # print("\n--- Bắt đầu Hàm 3: Lọc instances theo captions cuối cùng ---")
    # filter_instances_by_captions(
    #     instances_json_path=SYNCED_INSTANCES_PATH,
    #     filtered_captions_json_path=FINAL_CAPTIONS_PATH,
    #     output_json_path=FINAL_INSTANCES_PATH
    # )
    
    # # --- HÀM 4: Sao chép ảnh từ thư mục GỐC sang thư mục LỌC ---
    # print("\n--- Bắt đầu Hàm 4: Sao chép ảnh sketch đã lọc ---")
    # copy_filtered_images(
    #     filtered_json_path=FINAL_INSTANCES_PATH,
    #     source_image_dir=SOURCE_SKETCH_DIR,
    #     output_image_dir=OUTPUT_SKETCH_DIR,
    #     target_extension=SKETCH_FILE_EXTENSION
    # )
    
    # --- HÀM 5 (MỚI): Resize các ảnh đã lọc ---
    print("\n--- Bắt đầu Hàm 5: Resize ảnh về 256x256 ---")
    resize_images(
        source_dir=OUTPUT_SKETCH_DIR, # Lấy ảnh từ thư mục đã lọc
        output_dir=OUTPUT_SKETCH_RESIZED_DIR, # Lưu vào thư mục resize mới
        size=TARGET_IMAGE_SIZE,
        extension=SKETCH_FILE_EXTENSION
    )

    print("\n-------------------------------------------")
    print(f">>> ĐÃ HOÀN THÀNH TẤT CẢ! <<<")
    print(f"Dữ liệu JSON được lưu tại: {FILTERED_ANNOTATIONS_DIR}")
    print(f"Ảnh gốc đã lọc được lưu tại: {OUTPUT_SKETCH_DIR}")
    print(f"Ảnh resize 256x256 được lưu tại: {OUTPUT_SKETCH_RESIZED_DIR}")
    print(f"Thư mục tạm '{TEMP_DIR}' có thể được xóa.")