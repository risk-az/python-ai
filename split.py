import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image, ImageDraw
from shapely.geometry import shape, box, mapping, MultiPolygon, Polygon, MultiLineString, LineString, Point

def fix_invalid_geometry(geometry):
    if not geometry.is_valid:
        print("Геометрия невалидна. Исправляем...")
        # Попытка исправить геометрию
        geometry = geometry.buffer(0)  # Этот метод может исправить некоторые ошибки
    return geometry

def is_single_color(image_array):
    """Проверяет, состоит ли изображение из одного цвета"""
    return np.all(image_array == image_array[:, :, 0:1])  # Все пиксели одинаковые?

def get_bbox(x, y, tile_size, img_width, img_height):
    """Вычисляет BBOX в пикселях относительно оригинального изображения"""
    x_min = x * tile_size
    y_min = y * tile_size
    x_max = min(x_min + tile_size, img_width)  # Чтобы не выходить за границы
    y_max = min(y_min + tile_size, img_height)
    return [x_min, y_min, x_max, y_max]

def read_geojson(geojson_file):
    """Считывает GeoJSON и возвращает объект shapely (мультиполигон)"""
    with open(geojson_file, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    return fix_invalid_geometry(shape(geojson_data["features"][0]["geometry"]))

def save_label_studio(output_path, image_name, image_size, shapes):
    """Сохраняет данные в формате Label Studio"""
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
    """Разрезает GeoTIFF, вычисляет BBOX, проверяет пересечение с GeoJSON"""
    os.makedirs(output_folder, exist_ok=True)

    # Общий список аннотаций для Label Studio
    all_annotations = []

    all_names = []

    # Читаем мультиполигон из GeoJSON
    multipolygon = read_geojson(geojson_file)



    annotations_path = os.path.join(output_folder, "annotations.json")
    names_path= os.path.join(output_folder, "names.json")

    with rasterio.open(input_tif) as src:
        cols, rows = src.width, src.height  # Размеры исходного изображения
        grid_x = cols // tile_size  # Количество ячеек по ширине
        grid_y = rows // tile_size  # Количество ячеек по высоте

        total_tiles = grid_x * grid_y
        processed_tiles = 0  # Счётчик обработанных ячеек

        for y in range(grid_y):
            for x in range(grid_x):

                processed_tiles += 1
                # Вывод прогресса
                progress = (processed_tiles / total_tiles) * 100
                print(f"📝 Прогресс: {processed_tiles}/{total_tiles} ({progress:.2f}%)")

                window = Window(x * tile_size, y * tile_size, tile_size, tile_size)
                tile = src.read(window=window)

                # Преобразуем в массив (H, W, C)
                tile_rgb = np.moveaxis(tile[:3], 0, -1)  # Берем только первые 3 канала (RGB)

                if is_single_color(tile_rgb):
                    # print(f"🚫 Пропущен {x}_{y} (одноцветный)")
                    continue

                # Создаем PNG-изображение
                img = Image.fromarray(tile_rgb.astype(np.uint8))

                output_subfolder = os.path.join(output_folder, f"{tile_size}", f"{x}")

                url = f"{tile_size}/{x}/{tile_size}_{x}_{y}.png"

                # Имя файла
                output_filename = os.path.join(output_subfolder, f"{tile_size}_{x}_{y}.png")


                # Вычисляем BBOX
                bbox = get_bbox(x, y, tile_size, cols, rows)

                # Проверяем пересечение с GeoJSON
                bbox_polygon = box(*bbox)  # Преобразуем BBOX в shapely Polygon

                try:
                    intersection = multipolygon.intersection(bbox_polygon)
                except Exception  as e:
                    # Обработка исключения
                    print(f"Ошибка: {e}")
                    continue

                if intersection.is_empty:
                    # print(f"🚫 Пропущен {x}_{y} (нет пересечения)")
                    continue

                os.makedirs(output_subfolder, exist_ok=True)
                img.save(output_filename, format="PNG")


                # 🔹 Логируем WKT пересечения
                # print(f"🔍 Пересечение найдено для {x}_{y}: {intersection.wkt}")

                # Формируем данные для Label Studio
                shapes = []
                if intersection.geom_type == "Polygon":
                    coords = list(intersection.exterior.coords)
                    shapes.append({
                        "original_width": tile_size,
                        "original_height": tile_size,
                        "image_rotation": 0,
                        "value": {
                            "points": [[(x - bbox[0]) / tile_size * 100, (y - bbox[1]) / tile_size * 100] for x, y in
                                       coords],
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
                                "points": [[(x - bbox[0]) / tile_size * 100, (y - bbox[1]) / tile_size * 100] for x, y
                                           in coords],
                                "polygonlabels": ["Tracks"]
                            },
                            "id": f"polygon_{x}_{y}",
                            "from_name": "label",
                            "to_name": "image",
                            "type": "polygon"
                        })
                elif intersection.geom_type == "Point":
                    # print(f"⚠️ Пересечение для {x}_{y} - это точка, игнорируем.")
                    continue
                elif intersection.geom_type == "LineString":
                    # print(f"⚠️ Пересечение для {x}_{y} - это линия, игнорируем.")
                    continue

                mask = Image.new("L", (tile_size, tile_size), 0)  # Create blank mask
                draw = ImageDraw.Draw(mask)

                if intersection.geom_type == "Polygon":
                    coords =  [(px - bbox[0], py - bbox[1]) for px, py in intersection.exterior.coords]
                    # print(f"Mask: {coords}")
                    draw.polygon(coords, outline=255, fill=255)
                elif intersection.geom_type == "MultiPolygon":
                    for poly in intersection.geoms:
                        coords = [(px - bbox[0], py - bbox[1]) for px, py in poly.exterior.coords]
                        # print(f"Mask: {coords}")
                        draw.polygon(coords, outline=255, fill=255)

                mask_filename = os.path.join(output_subfolder, f"{tile_size}_{x}_{y}_m.png")
                mask.save(mask_filename)
                # print(f"Saved mask: {mask_filename}")

                annotation = {
                    "data": {"image": f"{server_url}/{url}"},
                    "annotations": [{
                        "result": shapes,
                        "meta": {"image_size": (tile_size, tile_size)}
                    }]
                }

                all_annotations.append(annotation)

                all_names.append(f"{tile_size}/{x}/{tile_size}_{x}_{y}")

                # print(f"✅ Сохранен {output_filename}")

                # with open(annotations_path, "w", encoding="utf-8") as f:
                #     json.dump(all_annotations, f, ensure_ascii=False, indent=4)

                # return



    # Сохраняем в формате Label Studio

    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(all_names, f, ensure_ascii=False, indent=4)

    print(f"📂 Все имена сохранены в {names_path}")

    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=4)

    print(f"📂 Все аннотации сохранены в {annotations_path}")


            # Использование
input_tif = "result.tif"
geojson_file = "pixel2.geojson"
output_folder = "output"
split_tif_to_png(input_tif, geojson_file, output_folder, 256)

print("🎉 Разрезка завершена!")
