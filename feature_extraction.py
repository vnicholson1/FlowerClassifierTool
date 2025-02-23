from PIL import Image, ImageFilter
import numpy as np
import cv2


def best_feature_extraction(pillow_image):
    smaller_image = pillow_image.resize([4,4], Image.Resampling.LANCZOS)
    tiny_image = np.array(smaller_image).flatten()
    greyscale = pillow_image.convert('L')
    edge_image = greyscale.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.EDGE_ENHANCE_MORE).filter(ImageFilter.FIND_EDGES)
    smaller_image = edge_image.resize([8,8], Image.Resampling.LANCZOS)
    # edge_image.show()
    # pillow_image.show()
    flattened_edges = np.array(smaller_image).flatten()
    edge_and_colours = np.concatenate((tiny_image, flattened_edges))
    return edge_and_colours



def get_sift_features(image_path, num_features):
    array_length = num_features * 128
    img = cv2.imread(image_path)
    img = cv2.resize(img,(150,150))
    # Applying SIFT detector
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(num_features)
    kp, des = sift.detectAndCompute(gray,None)
    # img=cv2.drawKeypoints(gray,kp,img)
    # cv2.imwrite('image-with-keypoints.jpg',img)

    hist, _ = np.histogram(des, bins=50)
    return hist


def to_tiny_image(image_path, image_size):
    pillow_image = Image.open(image_path)
    smaller_image = pillow_image.resize(image_size, Image.Resampling.LANCZOS)
    # array_to_image = Image.fromarray(image_array)
    # array_to_image.show()
    return np.array(smaller_image).flatten()


def to_tiny_image_with_edges(image_path, image_size):
    pillow_image = Image.open(image_path)
    greyscale = pillow_image.convert('L')
    edge_image = greyscale.filter(ImageFilter.SMOOTH_MORE).filter(ImageFilter.EDGE_ENHANCE_MORE).filter(ImageFilter.FIND_EDGES)
    smaller_image = edge_image.resize(image_size, Image.Resampling.LANCZOS)
    # edge_image.show()
    # pillow_image.show()
    flattened_edges = np.array(smaller_image).flatten()
    return flattened_edges


def to_edge_and_colour(image_path, tiny_image_size, edge_image_size):
    edge_and_colours = np.concatenate((to_tiny_image(image_path, tiny_image_size), to_tiny_image_with_edges(image_path, edge_image_size)))
    return edge_and_colours


def to_colour_histogram(image_path, num_bins):
    pillow_image = Image.open(image_path)
    red, green, blue = pillow_image.split()
    red_hist, _ = np.histogram(red, bins=num_bins)
    green_hist, _ = np.histogram(green, bins=num_bins)
    blue_hist, _ = np.histogram(blue, bins=num_bins)
    return np.concatenate((red_hist, green_hist, blue_hist))


def to_tiny_image_then_colour_histogram(image_path, num_bins):
    pillow_image = Image.open(image_path)
    pillow_image = pillow_image.resize((16,16), Image.Resampling.LANCZOS)
    red, green, blue = pillow_image.split()
    red_hist, _ = np.histogram(red, bins=num_bins)
    green_hist, _ = np.histogram(green, bins=num_bins)
    blue_hist, _ = np.histogram(blue, bins=num_bins)
    return np.concatenate((red_hist, green_hist, blue_hist))
