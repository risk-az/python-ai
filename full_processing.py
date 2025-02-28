import os
import gc
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

def is_single_color(image_array):
    """
    Проверяем, одноцветный ли тайл (image_array: (H, W, 3)).
    Если все пиксели совпадают по всем каналам — возвращаем True.
    """
    return np.all(image_array == image_array[..., :1])

def preprocess_tile(tile_rgb):
    """
    Превращаем NumPy-массив (H, W, 3) в тензор (1,3,256,256) для модели.
    """
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    pil_img = Image.fromarray(tile_rgb)
    return transform(pil_img).unsqueeze(0).to(device)

def to_red_mask(output_tensor):
    """
    Из выхода модели (логиты, shape: (1,1,256,256)) делаем бинарную маску (0 или 255).
    Возвращаем 3-канальный массив (shape: (3, 256, 256)), где:
      - Красный канал = маска (0 или 255)
      - Зелёный канал = 0
      - Синий канал   = 0
    """
    # Сигмоида + порог
    output = torch.sigmoid(output_tensor).squeeze().cpu().numpy()  # (256, 256)
    bin_mask = (output > 0.5).astype(np.uint8) * 255  # 0 или 255

    # Формируем 3 канала
    red_channel = bin_mask
    green_channel = np.zeros_like(bin_mask, dtype=np.uint8)
    blue_channel = np.zeros_like(bin_mask, dtype=np.uint8)

    # Финальный массив: (3, 256, 256)
    return np.stack([red_channel, green_channel, blue_channel], axis=0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Пример: грузим модель DeepLabV3 с 1 выходным каналом
    model = models.segmentation.deeplabv3_resnet101(weights=None, num_classes=1)
    checkpoint = torch.load("deeplabv3_model_epoch_300.pth", map_location=device)
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if "aux_classifier" not in k}
    model.load_state_dict(filtered_checkpoint, strict=False)
    model.to(device)
    model.eval()

    input_folder = "./input"
    output_folder = "./output"
    os.makedirs(output_folder, exist_ok=True)

    tile_size = 256  # Размер тайла, с которым будем читать

    for tif_file in sorted(os.listdir(input_folder)):
        if tif_file.endswith(".tif"):
            base_name = os.path.splitext(tif_file)[0]
            tif_path = os.path.join(input_folder, tif_file)

            # Папка для конкретного TIFF
            out_dir = os.path.join(output_folder, base_name)
            os.makedirs(out_dir, exist_ok=True)

            # Итоговый PNG: output/tif_name/tif_name_mask.png
            out_png_path = os.path.join(out_dir, f"{base_name}_mask.png")

            print(f"\nОбработка {tif_file} -> {out_png_path}")

            # Открываем входной TIFF
            with rasterio.open(tif_path) as src:
                cols, rows = src.width, src.height

                # Создаём большой итоговый массив (rows, cols, 3), где будет красная маска
                # По умолчанию чёрный фон (все нули)
                final_array = np.zeros((rows, cols, 3), dtype=np.uint8)

                # Вычисляем, сколько тайлов по X и Y
                grid_x = (cols + tile_size - 1) // tile_size
                grid_y = (rows + tile_size - 1) // tile_size

                total_tiles = grid_x * grid_y
                with tqdm(total=total_tiles, desc="Обработка тайлов", leave=True) as pbar:
                    for ty in range(grid_y):
                        for tx in range(grid_x):
                            x_min = tx * tile_size
                            y_min = ty * tile_size
                            x_max = min(x_min + tile_size, cols)
                            y_max = min(y_min + tile_size, rows)

                            # Читаем окно (window) из исходного TIFF
                            window = Window(x_min, y_min,
                                            x_max - x_min,
                                            y_max - y_min)
                            tile = src.read([1, 2, 3], window=window)  # (3, H, W)
                            tile_rgb = np.moveaxis(tile, 0, -1)        # (H, W, 3)

                            # Проверка одноцветности
                            if is_single_color(tile_rgb):
                                # Если тайл одноцветный — пропускаем модель, оставляем чёрный
                                pbar.update(1)
                                continue

                            # Преобразуем в тензор
                            tile_tensor = preprocess_tile(tile_rgb)

                            # Прогоняем через модель
                            with torch.no_grad():
                                output = model(tile_tensor)["out"]  # (1,1,256,256)

                            # Получаем 3-канальную красную маску
                            mask_3ch = to_red_mask(output)  # (3, 256, 256)

                            # На границах изображение может быть меньше 256×256 — обрежем маску
                            tile_width = x_max - x_min
                            tile_height = y_max - y_min
                            if tile_width < tile_size or tile_height < tile_size:
                                mask_3ch = mask_3ch[:, :tile_height, :tile_width]

                            # mask_3ch: (3, tileH, tileW)
                            # Преобразуем в (tileH, tileW, 3), чтобы вставить в final_array
                            mask_3ch_hw = np.moveaxis(mask_3ch, 0, -1)  # (tileH, tileW, 3)

                            # Вставляем маску в итоговый массив
                            final_array[y_min:y_max, x_min:x_max, :] = mask_3ch_hw

                            # Освобождаем память
                            del tile, tile_rgb, tile_tensor, output, mask_3ch, mask_3ch_hw
                            gc.collect()
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()

                            pbar.update(1)

            # Когда все тайлы обработаны, сохраняем итог в PNG
            final_img = Image.fromarray(final_array, mode='RGB')
            final_img.save(out_png_path)
            print(f"✅ PNG сохранён: {out_png_path}")

    print("\n🎉 Все файлы успешно обработаны!")
