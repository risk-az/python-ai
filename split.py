import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image, ImageDraw
from shapely.geometry import shape, box, mapping, MultiPolygon, Polygon, MultiLineString, LineString, Point


def fix_invalid_geometry(geometry):
    """Исправляет невалидную геометрию."""
    if not geometry.is_valid:
        print("Геометрия невалидна. Исправляем...")
        geometry = geometry.buffer(0)  # Исправление геометрии
    return geometry


def is_single_color(image_array):
    """Проверяет, состоит ли изображение из одного цвета."""
    return np.all(image_array == image_array[:, :, 0:1])


def get_bbox(x, y, tile_size, img_width, img_height):
    """Вычисляет BBOX в пикселях относительно оригинального изображения."""
    x_min = x * tile_size
    y_min = y * tile_size
    x_max = min(x_min + tile_size, img_width)  # Чтобы не выходить за границы
    y_max = min(y_min + tile_size, img_height)
    return [x_min, y_min, x_max, y_max]


def read_geojson(geojson_file):
    """Считывает GeoJSON и возвращает объект shapely (мультиполигон)."""
    with open(geojson_file, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    return fix_invalid_geometry(shape(geojson_data["features"][0]["geometry"]))


def save_label_studio(output_path, image_name, image_size, shapes):
    """Сохраняет данные в формате Label Studio."""
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
    """Разрезает GeoTIFF, вычисляет BBOX, проверяет пересечение с GeoJSON."""
    os.makedirs(output_folder, exist_ok=True)

    # Общий список аннотаций для Label Studio
    all_annotations = []
    all_names = []

    # Читаем мультиполигон из GeoJSON
    multipolygon = read_geojson(geojson_file)

    annotations_path = os.path.join(output_folder, "annotations.json")
    names_path = os.path.join(output_folder, "names.json")

    with rasterio.open(input_tif) as src:
        cols, rows = src.width, src.height  # Размеры исходного изображения
        grid_x = (cols + tile_size - 1) // tile_size  # Количество ячеек по ширине
        grid_y = (rows + tile_size - 1) // tile_size  # Количество ячеек по высоте

        total_tiles = grid_x * grid_y
        processed_tiles = 0  # Счётчик обработанных ячеек

        for y in range(grid_y):
            for x in range(grid_x):
                processed_tiles += 1
                # Вывод прогресса
                progress = (processed_tiles / total_tiles) * 100
                print(f"📝 Прогресс: {processed_tiles}/{total_tiles} ({progress:.2f}%)")

                # Вычисляем BBOX
                x_min = x * tile_size
                y_min = y * tile_size
                x_max = min(x_min + tile_size, cols)
                y_max = min(y_min + tile_size, rows)

                # Создаем прозрачное изображение 256x256
                img = Image.new("RGBA", (tile_size, tile_size), (0, 0, 0, 0))

                # Читаем фрагмент исходного изображения
                window = Window(x_min, y_min, x_max - x_min, y_max - y_min)
                tile = src.read(window=window)

                # Преобразуем в массив (H, W, C)
                tile_rgb = np.moveaxis(tile[:3], 0, -1)  # Берем только первые 3 канала (RGB)

                if is_single_color(tile_rgb):
                    continue  # Пропускаем одноцветные тайлы

                # Вставляем фрагмент в прозрачное изображение
                fragment = Image.fromarray(tile_rgb.astype(np.uint8))
                img.paste(fragment, (0, 0))

                # Сохраняем тайл
                output_subfolder = os.path.join(output_folder, f"{tile_size}", f"{x}")
                os.makedirs(output_subfolder, exist_ok=True)
                output_filename = os.path.join(output_subfolder, f"{tile_size}_{x}_{y}.png")
                img.save(output_filename, format="PNG")

                # Вычисляем BBOX для проверки пересечения
                bbox = get_bbox(x, y, tile_size, cols, rows)
                bbox_polygon = box(*bbox)  # Преобразуем BBOX в shapely Polygon

                try:
                    intersection = multipolygon.intersection(bbox_polygon)
                except Exception as e:
                    print(f"Ошибка: {e}")
                    continue

                if intersection.is_empty:
                    continue  # Пропускаем тайлы без пересечения

                # Формируем данные для Label Studio
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

                # Создаем маску
                mask = Image.new("L", (tile_size, tile_size), 0)
                draw = ImageDraw.Draw(mask)
                if intersection.geom_type == "Polygon":
                    coords = [(px - bbox[0], py - bbox[1]) for px, py in intersection.exterior.coords]
                    draw.polygon(coords, outline=255, fill=255)
                elif intersection.geom_type == "MultiPolygon":
                    for poly in intersection.geoms:
                        coords = [(px - bbox[0], py - bbox[1]) for px, py in poly.exterior.coords]
                        draw.polygon(coords, outline=255, fill=255)

                # Сохраняем маску
                mask_filename = os.path.join(output_subfolder, f"{tile_size}_{x}_{y}_m.png")
                mask.save(mask_filename)

                # Формируем аннотацию для Label Studio
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

    # Сохраняем все аннотации и имена
    with open(names_path, "w", encoding="utf-8") as f:
        json.dump(all_names, f, ensure_ascii=False, indent=4)

    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, ensure_ascii=False, indent=4)

    print(f"📂 Все имена сохранены в {names_path}")
    print(f"📂 Все аннотации сохранены в {annotations_path}")


# Использование
input_tif = "1000.tif"
geojson_file = "output_pixels.geojson"
output_folder = "output"
split_tif_to_png(input_tif, geojson_file, output_folder, 256)

print("🎉 Разрезка завершена!")