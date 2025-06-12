# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import hog

# Функція для завантаження зображень
def load_image(path):
    return np.array(Image.open(path).convert('L'))  # Конвертація в градаціях сірого

# Функція для масштабування зображення до фіксованого розміру 60×60
def resize_to_fixed(image):
    return cv2.resize(image, (60, 60), interpolation=cv2.INTER_AREA)

# Функція для адаптивного гістограмного вирівнювання (CLAHE)
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)

# Функція для виділення країв методом Собеля [3×3]
def apply_sobel(image):
    if image.ndim > 2:
        raise ValueError("Функція apply_sobel працює лише з одноканальними зображеннями.")
    Gx = np.array( [[-1, 0, 1],
                    [-2, 0, 2], 
                    [-1, 0, 1]])
    Gy = np.array( [[-1, -2, -1], 
                    [0, 0, 0], 
                    [1, 2, 1]])
    
    sobel_x = cv2.filter2D(image, cv2.CV_64F, Gx)
    sobel_y = cv2.filter2D(image, cv2.CV_64F, Gy)
    
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return sobel_magnitude.astype(np.uint8)

# Нормалізована кореляція
def compute_normalized_correlation(image, template):
    h, w = image.shape
    t_h, t_w = template.shape[:2]
    result = np.zeros((h - t_h + 1, w - t_w + 1), dtype=np.float64)
    w_mean = np.mean(template)
    w_diff = template - w_mean
    w_denominator = np.sqrt(np.sum(w_diff**2))
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            region = image[y:y+t_h, x:x+t_w]
            f_mean = np.mean(region)
            f_diff = region - f_mean
            numerator = np.sum(f_diff * w_diff)
            f_denominator = np.sqrt(np.sum(f_diff**2))
            denominator = f_denominator * w_denominator
            if denominator == 0:
                result[y, x] = 0
            else:
                result[y, x] = numerator / denominator
    return result

# Функція для обчислення центральних моментів
def calculate_central_moments(image):
    h, w = image.shape[:2]
    m00 = np.sum(image)
    if m00 == 0:
        return np.zeros((4, 4)), 0
    
    # Обчислення центроїда
    x_c = np.sum(np.arange(w) * np.sum(image, axis=0)) / m00
    y_c = np.sum(np.arange(h) * np.sum(image, axis=1)) / m00
    
    # Обчислення центральних моментів до порядку 3
    mu = np.zeros((4, 4))
    for p in range(4):
        for q in range(4):
            if p + q <= 3:
                mu[p, q] = np.sum(image * ((np.arange(h)[:, np.newaxis] - y_c) ** q) * ((np.arange(w) - x_c) ** p))
    return mu, m00

# Функція для обчислення нормалізованих моментів
def calculate_normalized_moments(mu, m00):
    eta = np.zeros((4, 4))
    for p in range(4):
        for q in range(4):
            if p + q >= 2 and p + q <= 3 and m00 != 0:
                eta[p, q] = mu[p, q] / (m00 ** ((p + q) / 2 + 1))
    return eta

# Функція для обчислення Hu-інваріантів
def compute_hu_moments(image):
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    # Беремо логарифм для стабільності
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-9)
    return hu_moments.flatten()[:4]

# Функція для обчислення інваріантних моментів
def compute_invariant_moments(image):
    mu, m00 = calculate_central_moments(image)
    eta = calculate_normalized_moments(mu, m00)
    
    phi1 = eta[2, 0] + eta[0, 2]
    phi2 = (eta[2, 0] - eta[0, 2])**2 + 4 * eta[1, 1]**2
    phi3 = (eta[3, 0] - 3 * eta[1, 2])**2 + (3 * eta[2, 1] - eta[0, 3])**2
    phi4 = (eta[3, 0] + eta[1, 2])**2 + (eta[2, 1] + eta[0, 3])**2
    
    hu_moments = compute_hu_moments(image)
    return np.concatenate([np.array([phi1, phi2, phi3, phi4]), hu_moments])

# Функція для обчислення HOG-ознак із фіксованим розміром
def compute_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    image_resized = resize_to_fixed(image)
    return hog(image_resized, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, visualize=False, feature_vector=True)

# Функція для обчислення геометричних ознак
def compute_geometric_features(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        compactness = (perimeter ** 2) / area if area > 0 else 0
    else:
        aspect_ratio = 0
        compactness = 0
    return aspect_ratio, compactness

# Функція для класифікації об’єкта
def classify_object(features, ref_military, ref_civilian, threshold=2000.0):
    moments, aspect_ratio, compactness, hog_features = features
    ref_military_moments, ref_military_aspect, ref_military_compactness, ref_military_hog = ref_military
    ref_civilian_moments, ref_civilian_aspect, ref_civilian_compactness, ref_civilian_hog = ref_civilian
    
    # Нормалізація моментів
    epsilon = 1e-9
    norm_moments = np.array([1.0] + [moments[i]/(moments[0] + epsilon) for i in range(1, len(moments))])
    norm_ref_military = np.array([1.0] + [ref_military_moments[i]/(ref_military_moments[0] + epsilon) for i in range(1, len(ref_military_moments))])
    norm_ref_civilian = np.array([1.0] + [ref_civilian_moments[i]/(ref_civilian_moments[0] + epsilon) for i in range(1, len(ref_civilian_moments))])
    
    # Вагові коефіцієнти для моментів
    weights_moments = np.array([0.1, 0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1])
    
    # Обчислюємо дистанцію для моментів
    dist_military_moments = np.sqrt(np.sum(weights_moments * (norm_moments - norm_ref_military)**2))
    dist_civilian_moments = np.sqrt(np.sum(weights_moments * (norm_moments - norm_ref_civilian)**2))
    
    # Обчислюємо дистанцію для HOG-ознак
    dist_military_hog = np.sqrt(np.sum((hog_features - ref_military_hog)**2)) if len(hog_features) == len(ref_military_hog) else 0.0
    dist_civilian_hog = np.sqrt(np.sum((hog_features - ref_civilian_hog)**2)) if len(hog_features) == len(ref_civilian_hog) else 0.0
    
    # Обчислюємо дистанцію для геометричних ознак
    dist_military_geo = np.sqrt((aspect_ratio - ref_military_aspect)**2 * 0.5 + (compactness - ref_military_compactness)**2 * 0.5)
    dist_civilian_geo = np.sqrt((aspect_ratio - ref_civilian_aspect)**2 + (compactness - ref_civilian_compactness)**2)
    
    # Асиметрія
    asymmetry = norm_moments[4]
    
    # Зміна ваги HOG залежно від асиметрії
    if 400 < asymmetry < 600:
        weights_hog = 0.5
        weights_moments_total = 0.3
        weights_geo = 0.2
    else:
        weights_hog = 0.3
        weights_moments_total = 0.5
        weights_geo = 0.2
    
    # Комбінуємо дистанції
    dist_military = weights_moments_total * dist_military_moments + weights_hog * dist_military_hog + weights_geo * dist_military_geo
    dist_civilian = weights_moments_total * dist_civilian_moments + weights_hog * dist_civilian_hog + weights_geo * dist_civilian_geo
    
    print(f"Дистанція до військового: {dist_military:.4f}, до цивільного: {dist_civilian:.4f}")
    print(f"HOG до військового: {dist_military_hog:.4f}, HOG до цивільного: {dist_civilian_hog:.4f}")
    print(f"Аспектне відношення: {aspect_ratio:.2f}, Компактність: {compactness:.2f}")
    print(f"Нормалізовані моменти об’єкта: {norm_moments}")
    
    # Пріоритет за асиметрією
    if asymmetry > 600:
        return "Військовий", dist_military
    
    # Пріоритет HOG для асиметрії 400–600
    if 400 < asymmetry < 600 and dist_military_hog < dist_civilian_hog * 0.8:
        return "Військовий", dist_military
    
    # Поріг для цивільних
    if asymmetry < 500 and dist_civilian < dist_military * 0.9:
        return "Цивільний", dist_civilian
    
    # Відносний поріг для класифікації
    if dist_military < dist_civilian * 0.7 and dist_military < threshold:
        return "Військовий", dist_military
    elif dist_civilian < dist_military * 0.7 and dist_civilian < threshold:
        return "Цивільний", dist_civilian
    else:
        return "Невідомий", min(dist_military, dist_civilian)

# Шляхи до зображень
image_path = r"C:\Users\kaple\OneDrive\Рабочий стол\DIP\original00.jpg"
template_path = r"C:\Users\kaple\OneDrive\Рабочий стол\DIP\imba.jpg"
military_template_path = r"C:\Users\kaple\OneDrive\Рабочий стол\DIP\pon.jpg"
civilian_template_path = r"C:\Users\kaple\OneDrive\Рабочий стол\DIP\wo.jpg"

# Завантаження зображень
original_image = load_image(image_path)
template = load_image(template_path)
military_template = load_image(military_template_path)
civilian_template = load_image(civilian_template_path)

# Зберігаємо початкове зображення для відображення рамок
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

# Обробка початкового зображення
processed_image = cv2.GaussianBlur(original_image, (5, 5), 0)
processed_image = apply_clahe(processed_image, clip_limit=1.5, tile_grid_size=(8, 8))
processed_image = apply_sobel(processed_image)
kernel = np.ones((3, 3), np.uint8)
processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, kernel)
processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Обробка еталона для пошуку
template = apply_clahe(template, clip_limit=2.0, tile_grid_size=(8, 8))
template = apply_sobel(template)
template = resize_to_fixed(template)

# Обчислення довжини контуру еталона
_, binary_template = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_template, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    template_contour_length = cv2.arcLength(largest_contour, True)
    print(f"Довжина контуру еталона: {template_contour_length:.2f}")
else:
    raise ValueError("Контур еталона не знайдено.")

# Обробка еталонів для класифікації
military_template = resize_to_fixed(military_template)
civilian_template = resize_to_fixed(civilian_template)

sobel_military = apply_sobel(military_template)
sobel_civilian = apply_sobel(civilian_template)

reference_military_moments = compute_invariant_moments(sobel_military)
reference_civilian_moments = compute_invariant_moments(sobel_civilian)
reference_military_aspect, reference_military_compactness = compute_geometric_features(sobel_military)
reference_civilian_aspect, reference_civilian_compactness = compute_geometric_features(sobel_civilian)
reference_military_hog = compute_hog_features(sobel_military)
reference_civilian_hog = compute_hog_features(sobel_civilian)

reference_military = (reference_military_moments, reference_military_aspect, reference_military_compactness, reference_military_hog)
reference_civilian = (reference_civilian_moments, reference_civilian_aspect, reference_civilian_compactness, reference_civilian_hog)

print(f"Ознаки військового шаблону: Моменти={reference_military_moments}, Аспектне відношення={reference_military_aspect:.2f}, Компактність={reference_military_compactness:.2f}")
print(f"Ознаки цивільного шаблону: Моменти={reference_civilian_moments}, Аспектне відношення={reference_civilian_aspect:.2f}, Компактність={reference_civilian_compactness:.2f}")

# Відображення обробленого зображення
plt.figure(figsize=(10, 8))
plt.imshow(processed_image, cmap='gray')
plt.title("Оброблене зображення (Gaussian Blur + CLAHE + Собель + Morph Opening)")
plt.axis('off')
plt.show()

# Відображення обробленого еталона
plt.figure(figsize=(5, 5))
plt.imshow(template, cmap='gray')
plt.title("Оброблений еталон (CLAHE + Собель)")
plt.axis('off')
plt.show()

# Обчислення моментів еталона для пошуку
template_features = compute_invariant_moments(binary_template)
print(f"Ознаки еталона для пошуку: Моменти={template_features}")

# Розширення початкового зображення
t_h, t_w = template.shape[:2]
pad_size = max(t_h // 2, t_w // 2)
padded_image = np.pad(processed_image, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)

# Перетворення зображень у float32 для кореляції
image_float = padded_image.astype(np.float32)
template_float = template.astype(np.float32)

# Обчислення нормалізованої кореляції
result = compute_normalized_correlation(image_float, template_float)

# Встановлення порогу та пошук локальних максимумів
threshold = 0.8
min_distance = 50

# Нормалізація карти кореляції до [0, 1]
result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)

# Знаходимо локальні максимуми
dilated = cv2.dilate(result, np.ones((min_distance, min_distance)))
maxima = (result == dilated) & (result > threshold)
y, x = np.where(maxima)
peaks = list(zip(x, y))

# Відображення карти кореляції
plt.figure(figsize=(10, 8))
plt.imshow(result, cmap='gray')
if peaks:
    x_peaks, y_peaks = zip(*peaks)
    
    plt.legend()
plt.title("Карта кореляції з локальними максимумами")
plt.axis('off')
plt.show()

# Фільтрація піків
filtered_peaks = []
peak_info = []
objects_for_invariants = []
invariant_counter = 0

for i, (x, y) in enumerate(peaks):
    adj_x = x + t_w // 2 - pad_size
    adj_y = y + t_h // 2 - pad_size
    if 0 <= adj_x < original_image.shape[1] and 0 <= adj_y < original_image.shape[0]:
        top_left_x = max(0, adj_x - t_w // 2)
        top_left_y = max(0, adj_y - t_h // 2)
        bottom_right_x = min(original_image.shape[1], adj_x + t_w // 2)
        bottom_right_y = min(original_image.shape[0], adj_y + t_h // 2)
        
        # Вирізаємо регіон для аналізу
        region = original_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x].copy()
        
        # Використовуємо адаптивну бінаризацію для фільтрації
        binary_region = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        binary_region = cv2.morphologyEx(binary_region, cv2.MORPH_CLOSE, kernel)
        
        # Знаходимо контури
        contours, _ = cv2.findContours(binary_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            compactness = (perimeter ** 2) / area if perimeter > 0 else 0
            mean_intensity = np.mean(region)
            
            # Фільтри для відсіювання піків
            if (mean_intensity > 90 and 
                area > 2900 and 
                compactness <= 100):
                # Отримуємо габарити контуру
                contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(largest_contour)
                # Переводимо координати контуру в глобальні координати зображення
                global_x = top_left_x + contour_x
                global_y = top_left_y + contour_y
                # Додаємо відступ (padding) 10 пікселів
                padding = 10
                top_left_x = max(0, global_x - padding)
                top_left_y = max(0, global_y - padding)
                bottom_right_x = min(original_image.shape[1], global_x + contour_w + padding)
                bottom_right_y = min(original_image.shape[0], global_y + contour_h + padding)
                
                filtered_peaks.append((adj_x, adj_y))
                peak_info.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
                # Вирізаємо об’єкт із сирих даних
                obj = original_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
                # Застосовуємо Собель для об’єкта
                sobel_obj = apply_sobel(obj)
                # Обчислюємо HOG-ознаки
                hog_obj = compute_hog_features(sobel_obj)
                objects_for_invariants.append((sobel_obj, sobel_obj, hog_obj))
                invariant_counter += 1

# Вивід об’єктів, до яких застосовуються інваріантні ознаки
print(f"Кількість об’єктів, до яких застосовуються інваріантні ознаки: {invariant_counter}")
if invariant_counter > 0:
    cols = min(invariant_counter, 5)
    rows = (invariant_counter + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for i, (obj, _, hog_obj) in enumerate(objects_for_invariants, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(obj, cmap='gray')
        plt.title(f"Об’єкт {i}")
        plt.axis('off')
    plt.suptitle("Об’єкти, до яких застосовуються інваріантні ознаки")
    plt.show()


# Класифікація відфільтрованих об’єктів
detected_objects = []
classified_objects = []
distances = []

for (x, y), (top_left_x, top_left_y, bottom_right_x, bottom_right_y) in zip(filtered_peaks, peak_info):
    # Вирізаємо об’єкт із адаптивними координатами
    obj = original_image[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    
    # Застосовуємо матрицю Собеля
    sobel_obj = apply_sobel(obj)
    
    moments = compute_invariant_moments(sobel_obj)
    aspect_ratio, compactness = compute_geometric_features(sobel_obj)
    hog_features = compute_hog_features(sobel_obj)
    features = (moments, aspect_ratio, compactness, hog_features)
    
    label, distance = classify_object(features, reference_military, reference_civilian)
    
    # Визначаємо колір рамки залежно від класу об’єкта
    if label == "Військовий":
        color = (255, 165, 0)  # для військових
    elif label == "Цивільний":
        color = (0, 255, 0)    # для цивільних
    else:
        color = (0, 0, 255)    # для невідомих
    
    # Малюємо лише одну рамку з відповідним кольором і товщиною
    cv2.rectangle(original_image_rgb, (top_left_x, top_left_y),
                  (bottom_right_x, bottom_right_y), color, 2)
    
    detected_objects.append(obj)
    classified_objects.append(label)
    distances.append(distance)


# Відображення початкового зображення з рамками
plt.figure(figsize=(10, 8))
plt.imshow(original_image_rgb)
plt.title("Початкове зображення з виділеними об’єктами")
plt.axis('off')
plt.show()

# Відображення вирізаних об’єктів із результатами класифікації
num_objects = len(detected_objects)
if num_objects > 0:
    cols = min(num_objects, 5)
    rows = (num_objects + cols - 1) // cols
    plt.figure(figsize=(15, 3 * rows))
    for i, (obj, label, dist) in enumerate(zip(detected_objects, classified_objects, distances), 1):
        plt.subplot(rows, cols, i)
        plt.imshow(obj, cmap='gray')
        plt.title(f"Об’єкт {i}\n{label}")
        plt.axis('off')
    plt.suptitle("Вирізані об’єкти з результатами класифікації")
    plt.show()
else:
    print("Об’єктів не знайдено.")

# %%
from PIL import Image
import matplotlib.pyplot as plt

# Шлях до зображення
image_path = r"C:\Users\kaple\OneDrive\Рабочий стол\DIP\original00.jpg"

# Відкриваємо зображення
image = Image.open(image_path)

# Перетворюємо зображення в чорно-біле (градації сірого)
bw_image = image.convert("L")

# Відображаємо зображення
plt.imshow(bw_image, cmap="gray")
plt.axis("off")  # Прибираємо осі
plt.show()

# %%
from PIL import Image
import matplotlib.pyplot as plt

# Шляхи до зображень еталонів
military_template_path = r"C:\Users\kaple\OneDrive\Рабочий стол\DIP\pon.jpg"
civilian_template_path = r"C:\Users\kaple\OneDrive\Рабочий стол\DIP\wo.jpg"

# Завантаження та конвертація в градації сірого
military_template = Image.open(military_template_path).convert('L')
civilian_template = Image.open(civilian_template_path).convert('L')

# Відображення еталонів
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(military_template, cmap='gray')
plt.title("Військовий еталон")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(civilian_template, cmap='gray')
plt.title("Цивільний еталон")
plt.axis('off')

plt.suptitle("Чорно-білі еталонні зображення")
plt.show()


