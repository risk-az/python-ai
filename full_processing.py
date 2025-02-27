import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image, ImageDraw
from shapely.geometry import shape, box
import torch
import torchvision.transforms as T
import torchvision.models as models
import threading

# Функции из split.py
def fix_invalid_geometry(geometry):
    if not geometry.is_valid:
        print("Геометрия невалидна. Исправляем...")
        geometry = geometry.buffer(0)
    return geometry

def is_single_color(image_array):
    return np.all(image_array == image_array[:, :, 0:1])

def get_bbox(x, y, tile_size, img_width, img_height):
    x_min = x * tile_size
    y_min = y * tile_size
    x_max = min(x_min + tile_size, img_width)
    y_max = min(y_min + tile_size, img_height)
    return [x_min, y_min, x_max, y_max]

def read_geojson(geojson_file):
    with open(geojson_file, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    return fix_invalid_geometry(shape(geojson_data["features"][0]["geometry"]))

def save_label_studio(output_path, image_name, image_size, shapes):
    label_studio_data = [{
        "data": {"image": image_name},
        "annotations": [{
            "result": shapes,
            "meta": {"image_size": image_size}
        }]
    }]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(label_studio_data, f, ensure_ascii=False, indent=4)

def split_tif_to_png(input_tif, geojson_file, output_folder, tile_size=256, server_url="http://localhost:12345/media"):
    os.makedirs(output_folder, exist_ok=True)
    with rasterio.open(input_tif) as src:
        cols, rows = src.width, src.height
        original_size_path = os.path.join(output_folder, "original_size.json")
        with open(original_size_path, "w", encoding="utf-8") as f:
            json.dump({"width": cols, "height": rows}, f)
    
    all_annotations = []
    all_names = []
    multipolygon = read_geojson(geojson_file)
    
    annotations_path = os.path.join(output_folder, "annotations.json")
    names_path = os.path.join(output_folder, "names.json")
    
    with rasterio.open(input_tif) as src:
        cols, rows = src.width, src.height
        grid_x = (cols + tile_size - 1) // tile_size
        grid_y = (rows + tile_size - 1) // tile_size
        
        total_tiles = grid_x * grid_y
        processed_tiles = 0
        
        for y in range(grid_y):
            for x in range(grid_x):
                processed_tiles += 1
                progress = (processed_tiles / total_tiles) * 100
                print(f"📝 Прогресс: {processed_tiles}/{total_tiles} ({progress:.2f}%)")
                
                x_min = x * tile_size
                y_min = y * tile_size
                x_max = min(x_min + tile_size, cols)
                y_max = min(y_min + tile_size, rows)
                
                # Создаём прозрачное изображение
                img = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))
                window = Window(x_min, y_min, x_max - x_min, y_max - y_min)
                tile = src.read(window=window)
                tile_rgb = np.moveaxis(tile[:3], 0, -1)
                
                if is_single_color(tile_rgb):
                    continue
                
                fragment = Image.fromarray(tile_rgb.astype(np.uint8))
                img.paste(fragment, (0, 0))
                
                # Сохраняем тайл в структуру output_folder/tile_size/x/
                output_subfolder = os.path.join(output_folder, f"{tile_size}", f"{x}")
                os.makedirs(output_subfolder, exist_ok=True)
                output_filename = os.path.join(output_subfolder, f"{tile_size}_{x}_{y}.png")
                img.save(output_filename, format="PNG")
                
                bbox = get_bbox(x, y, tile_size, cols, rows)
                bbox_polygon = box(*bbox)
                
                try:
                    intersection = multipolygon.intersection(bbox_polygon)
                except Exception as e:
                    print(f"Ошибка: {e}")
                    continue
                
                if intersection.is_empty:
                    continue
                
                shapes = []
                if intersection.geom_type == "Polygon":
                    coords = list(intersection.exterior.coords)
                    shapes.append({
                        "original_width": tile_size,
                        "original_height": tile_size,
                        "image_rotation": 0,
                        "value": {
                            "points": [[(px - bbox[0]) / tile_size * 100, (py - bbox[1]) / tile_size * 100] for px, py in coords],
                            "polygonlabels": ["Building"]
                        },
                        "id": f"polygon_{x}_{y}",
                        "from_name": "label",
                        "to_name": "image",
                        "type": "polygon"
                    })
                elif intersection.geom_type == "MultiPolygon":
                    for poly in intersection.geoms:
                        coords = list(poly.exterior.coords)
                        shapes.append({
                            "original_width": tile_size,
                            "original_height": tile_size,
                            "image_rotation": 0,
                            "value": {
                                "points": [[(px - bbox[0]) / tile_size * 100, (py - bbox[1]) / tile_size * 100] for px, py in coords],
                                "polygonlabels": ["Building"]
                            },
                            "id": f"polygon_{x}_{y}",
                            "from_name": "label",
                            "to_name": "image",
                            "type": "polygon"
                        })
                
                # Создаём и сохраняем маску
                mask = Image.new("L", (tile_size, tile_size), 0)
                draw = ImageDraw.Draw(mask)
                if intersection.geom_type == "Polygon":
                    coords = [(px - bbox[0], py - bbox[1]) for px, py in intersection.exterior.coords]
                    draw.polygon(coords, outline=255, fill=255)
                elif intersection.geom_type == "MultiPolygon":
                    for poly in intersection.geoms:
                        coords = [(px - bbox[0], py - bbox[1]) for px, py in poly.exterior.coords]
                        draw.polygon(coords, outline=255, fill=255)
                
                mask_filename = os.path.join(output_subfolder, f"{tile_size}_{x}_{y}_m.png")
                mask.save(mask_filename)
                
                url = f"{tile_size}/{x}/{tile_size}_{x}_{y}.png"
                annotation = {
                    "data": {"image": f"{server_url}/{url}"},
                    "annotations": [{
                        "result": shapes,
                        "meta": {"image_size": (tile_size, tile_size)}
                    }]
                }
                all_annotations.append(annotation)
                all_names.append(f"{tile_size}/{x}/{tile_size}_{x}_{y}")
    
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(all_names, f, ensure_ascii=False, indent=4)
    
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=4)
    
    print(f"📂 Все имена сохранены в {names_path}")
    print(f"📂 Все аннотации сохранены в {annotations_path}")

# Функции из compare.py
def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB").resize((256, 256))
    return transform(image).unsqueeze(0).to(device)

def filter_red_regions(output):
    red_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    red_mask[:, :, 0] = (output * 255).astype(np.uint8)
    return red_mask

def save_blended_image(folder_path, image_name, output):
    save_path = os.path.join(folder_path, image_name.replace(".png", "_p.png"))
    filtered_output = filter_red_regions(output)
    Image.fromarray(filtered_output).resize((256,256), Image.LANCZOS).save(save_path)
    print(f"✅ Сохранено: {save_path}")

def process_images(root_folder):
    for subfolder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(folder_path):
            continue
       
        image_files = sorted([f for f in os.listdir(folder_path)
                              if f.endswith(".png") and not f.endswith("_m.png") and not f.endswith("_p.png")])
        
        for image_name in image_files:
            image_path = os.path.join(folder_path, image_name)
            image_tensor = preprocess_image(image_path)
            with torch.no_grad():
                output = model(image_tensor)["out"].to(device)
                output = torch.sigmoid(output).squeeze().cpu().numpy()
                output = (output > 0.5).astype(np.uint8)
            save_blended_image(folder_path, image_name, output)

# Функции из union.py
def stitch_images_for_output(output_folder, root_folder):
    image_tiles = []
    for subfolder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(folder_path):
            continue
        
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith("_p.png"):
                parts = filename.replace("_p.png", "").split("_")
                if len(parts) == 3:
                    x, y = int(parts[1]), int(parts[2])
                    image_tiles.append((x, y, os.path.join(folder_path, filename)))
    
    if not image_tiles:
        print("⚠️ Нет изображений для объединения!")
        return
    
    original_size_path = os.path.join(output_folder, "original_size.json")
    with open(original_size_path, "r", encoding="utf-8") as f:
        original_size = json.load(f)
    original_width, original_height = original_size["width"], original_size["height"]
    
    stitched_image = Image.new("RGB", (original_width, original_height))
    
    for x, y, path in image_tiles:
        img = Image.open(path)
        paste_x = x * 256
        paste_y = y * 256
        if paste_x + 256 > original_width:
            img = img.crop((0, 0, original_width - paste_x, 256))
        if paste_y + 256 > original_height:
            img = img.crop((0, 0, 256, original_height - paste_y))
        stitched_image.paste(img, (paste_x, paste_y))
    
    output_path = os.path.join(root_folder, "stitched_output.png")
    stitched_image.save(output_path)
    print(f"✅ Большое изображение сохранено: {output_path}")

# Основной код
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Используется устройство: {device}")
    
    model = models.segmentation.deeplabv3_resnet101(weights=None, num_classes=1)
    checkpoint = torch.load("deeplabv3_model_epoch_260.pth", map_location=device)
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if "aux_classifier" not in k}
    model.load_state_dict(filtered_checkpoint, strict=False)
    model.to(device)
    model.eval()
    
    input_folder = "./input"
    # GeoJSON можно использовать общий или под каждый tif (при необходимости доработать)
    geojson_file = "output_pixels.geojson"
    
    for tif_file in sorted(os.listdir(input_folder)):
        if tif_file.endswith(".tif"):
            base_name = os.path.splitext(tif_file)[0]
            tif_path = os.path.join(input_folder, tif_file)
            output_folder = os.path.join("output", base_name)
            os.makedirs(output_folder, exist_ok=True)
            
            print(f"\nОбработка файла {tif_file}. Выходные данные будут в {output_folder}")
            
            # Этап split – создаём тайлы и сохраняем original_size.json в output/<base_name>
            split_tif_to_png(tif_path, geojson_file, output_folder, tile_size=256)
            
            # Определяем корневую папку для этапа compare: output/<base_name>/256
            current_root = os.path.join(output_folder, "256")
            
            # Этап compare – обработка изображений (с использованием потока)
            thread = threading.Thread(target=process_images, args=(current_root,))
            thread.start()
            thread.join()
            
            # Этап union – объединяем обработанные тайлы
            stitch_images_for_output(output_folder, current_root)
            
            print(f"🎉 Этапы для {tif_file} завершены!")
    
    print("\n🎉 Все этапы для всех файлов завершены!")
