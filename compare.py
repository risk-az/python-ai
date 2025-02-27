import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import threading

# Проверяем доступность GPU (NVIDIA) или CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Используется устройство: {device}")

# Главная папка с изображениями
root_folder = "./output/256"

# Загружаем модель
model = models.segmentation.deeplabv3_resnet101(weights=None, num_classes=1)
checkpoint = torch.load("deeplabv3_model_epoch_260.pth", map_location=device)
filtered_checkpoint = {k: v for k, v in checkpoint.items() if "aux_classifier" not in k}
model.load_state_dict(filtered_checkpoint, strict=False)
model.to(device)
model.eval()

# Предобработка изображения
def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize((256, 256)),  # Размер, который использовался при обучении
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB").resize((256, 256))  # Принудительно изменяем размер
    return transform(image).unsqueeze(0).to(device)  # Добавляем batch dimension и отправляем на GPU/CPU

# Фильтрация только красных областей
def filter_red_regions(output):
    red_mask = np.zeros((256, 256, 3), dtype=np.uint8)  # Создаем пустую маску RGB
    red_mask[:, :, 0] = (output * 255).astype(np.uint8)  # Красный канал
    return red_mask

# Функция сохранения изображения
def save_blended_image(folder_path, image_name, output):
    save_path = os.path.join(folder_path, image_name.replace(".png", "_p.png"))
    
    filtered_output = filter_red_regions(output)  # Оставляем только красные области
    
    Image.fromarray(filtered_output).resize((256,256),Image.LANCZOS).save(save_path);
    #fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)  # Пропорции для 256x256
    #ax.imshow(filtered_output)
    #ax.axis("off")
    #plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)
    #plt.close(fig)
    
    print(f"✅ Сохранено: {save_path}")

# Функция обработки всех изображений во всех папках в фоне
def process_images():
    for subfolder in sorted(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, subfolder)
        if not os.path.isdir(folder_path):
            continue
       
        
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png") and not f.endswith("_m.png") and not f.endswith("_p.png")])
        
        for image_name in image_files:
            image_path = os.path.join(folder_path, image_name)
            
            # Загружаем изображение
            image_tensor = preprocess_image(image_path)

            # Предсказание
            with torch.no_grad():
                output = model(image_tensor)["out"].to(device)  # Форма (1, H, W)
                output = torch.sigmoid(output).squeeze().cpu().numpy()  # Убираем лишнюю размерность (H, W)
                output = (output > 0.5).astype(np.uint8)  # Бинаризуем

            # Автосохранение в фоне (сохраняем только красные следы)
            save_blended_image(folder_path, image_name, output)

# Запускаем обработку в фоне
thread = threading.Thread(target=process_images)
thread.start()
