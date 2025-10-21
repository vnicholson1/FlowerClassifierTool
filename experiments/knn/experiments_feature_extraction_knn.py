import os
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import tqdm
import pickle

# Feature extraction methods
# Colour features
def extract_color_histogram(img, bins=(8, 8, 8)):
    # Convert to RGB just in case
    img = img.convert('RGB')
    arr = np.array(img)
    hist = np.histogramdd(
        arr.reshape(-1, 3), bins=bins, range=[(0, 256), (0, 256), (0, 256)]
    )[0]
    hist = hist.flatten()
    hist = hist / np.sum(hist)  # Normalize
    return hist

# Texture features
def extract_lbp(img, bins=16):
    img = img.convert('L')
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)
    for i in range(1, h-1):
        for j in range(1, w-1):
            center = arr[i, j]
            code = 0
            code |= (arr[i-1, j-1] > center) << 7
            code |= (arr[i-1, j  ] > center) << 6
            code |= (arr[i-1, j+1] > center) << 5
            code |= (arr[i,   j+1] > center) << 4
            code |= (arr[i+1, j+1] > center) << 3
            code |= (arr[i+1, j  ] > center) << 2
            code |= (arr[i+1, j-1] > center) << 1
            code |= (arr[i,   j-1] > center) << 0
            lbp[i-1, j-1] = code
    hist, _ = np.histogram(lbp, bins=bins, range=(0, 256))
    hist = hist / np.sum(hist)
    return hist

# Shape/edges
def extract_hog(img, cell_size=16, bins=8):
    img = img.convert('L')
    arr = np.array(img, dtype=np.float32)
    gx = np.zeros_like(arr)
    gy = np.zeros_like(arr)
    gx[:, :-1] = np.diff(arr, n=1, axis=1)
    gy[:-1, :] = np.diff(arr, n=1, axis=0)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * (180 / np.pi) % 180
    h, w = arr.shape
    n_cells_x = w // cell_size
    n_cells_y = h // cell_size
    hog = []
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ang = ang[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            hist, _ = np.histogram(cell_ang, bins=bins, range=(0, 180), weights=cell_mag)
            hog.extend(hist)
    hog = np.array(hog)
    if np.sum(hog) > 0:
        hog = hog / np.sum(hog)
    return hog


def _extract_features(img):
    # Color
    color_feat = extract_color_histogram(img, bins=(8,8,8))
    # Texture
    lbp_feat = extract_lbp(img, bins=256)
    # Shape/edges
    hog_feat = extract_hog(img, cell_size=16, bins=8)
    # Concatenate all
    feat = np.concatenate([color_feat, hog_feat, lbp_feat])
    return feat


# Noise removal methods
def center_crop(img, crop_ratio=0.8, jitter=0.05):
    w, h = img.size
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    dw = int(w * jitter)
    dh = int(h * jitter)
    left = np.random.randint((w - new_w)//2 - dw, (w - new_w)//2 + dw + 1)
    top = np.random.randint((h - new_h)//2 - dh, (h - new_h)//2 + dh + 1)
    right = left + new_w
    bottom = top + new_h
    return img.crop((left, top, right, bottom))


def extract_features(X):
    features = []
    for img in tqdm.tqdm(X, desc="Extracting features"):
        img_cleaned = center_crop(img, crop_ratio=0.8, jitter=0)
        feat = _extract_features(img_cleaned)
        features.append(feat)
    return np.array(features)


def load_images_from_folder(folder, size=(64, 64)):
    X, y = [], []
    class_names = sorted(os.listdir(folder))
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            path = os.path.join(class_dir, fname)
            try:
                img = Image.open(path).resize(size)
                X.append(img)
                y.append(label)
            except Exception as e:
                print(f"Error loading {path}: {e}")
    return X, y, class_names


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Image feature extraction + kNN classification')
    parser.add_argument('--k', type=int, default=3, help='k for kNN')
    args = parser.parse_args()

    train_dir = os.path.join('data', 'train')
    test_dir = os.path.join('data', 'test')

    print(f"Loading training images from {train_dir}...")
    if os.path.exists(os.path.join('experiments', 'train_data.pkl')):
        with open(os.path.join('experiments', 'train_data.pkl'), 'rb') as f:
            X_train_imgs, y_train, class_names = pickle.load(f)
    else:
        X_train_imgs, y_train, class_names = load_images_from_folder(train_dir)
        with open(os.path.join('experiments', 'train_data.pkl'), 'wb') as f:
            pickle.dump((X_train_imgs, y_train, class_names), f)
    print(f"Loaded {len(X_train_imgs)} training images from {len(class_names)} classes.")

    print(f"Loading test images from {test_dir}...")
    if os.path.exists(os.path.join('experiments', 'test_data.pkl')):
        with open(os.path.join('experiments', 'test_data.pkl'), 'rb') as f:
            X_test_imgs, y_test, class_names = pickle.load(f)
    else:
        X_test_imgs, y_test, class_names = load_images_from_folder(test_dir)
        with open(os.path.join('experiments', 'test_data.pkl'), 'wb') as f:
            pickle.dump((X_test_imgs, y_test, class_names), f)
    print(f"Loaded {len(X_test_imgs)} test images from {len(class_names)} classes.")

    print(f"Extracting features ...")
    X_train = extract_features(X_train_imgs)
    X_test = extract_features(X_test_imgs)

    clf = KNeighborsClassifier(n_neighbors=args.k, weights='distance', metric='manhattan')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

    print(classification_report(y_test, y_pred, digits=3))


if __name__ == '__main__':
    # best - combo k=3, p=1, distance weighted
    main()
