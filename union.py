from PIL import Image
import os

# Главная папка с обработанными изображениями
root_folder = "./output/256"

def stitch_images():
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
    
    max_x = max(tile[0] for tile in image_tiles)
    max_y = max(tile[1] for tile in image_tiles)
    
    stitched_image = Image.new("RGB", ((max_x ) * 256, (max_y ) * 256))
    
    for x, y, path in image_tiles:
        img = Image.open(path)
        stitched_image.paste(img, ((x-1) * 256, (y-1) * 256))  # Инверсия Y для правильного порядка
    
    output_path = os.path.join(root_folder, "stitched_output.png")
    stitched_image.save(output_path)
    print(f"✅ Большое изображение сохранено: {output_path}")

if __name__ == "__main__":
    stitch_images()