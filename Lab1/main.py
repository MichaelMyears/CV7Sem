import cv2
import numpy as np
import sys


def circle_corners(src):
    dst = cv2.cornerHarris(src, 2, 3, 0.04)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    w, h = dst_norm.shape
    color = (255, 0, 0)
    for i in range(w):
        for j in range (h):
            if int(dst_norm[i][j]) > 60:
                cv2.circle(src, (j, i), 2, color)


def calculate_pixel_color(image, x, y, radius):
        w, h, _ = image.shape
        b_arr = []
        g_arr = []
        r_arr = []
        for j in range(-radius, radius + 1):
            for i in range(-radius, radius + 1):
                index_x = np.clip(x + j, 0, w - 1)
                index_y = np.clip(y + i, 0, h - 1)
                b_arr.append(image[index_x][index_y][0])
                g_arr.append(image[index_x][index_y][1])
                r_arr.append(image[index_x][index_y][2])
        new_color = np.array([np.median(b_arr), np.median(g_arr), np.median(r_arr)], dtype=np.int8)
        return new_color


def median_filter(src, distance):
    dst = src.copy()
    w, h, _ = src.shape
    for j in range(w):
        for i in range(h):
            dst[j][i] = calculate_pixel_color(src, j, i, int(0.2 * distance[j][i]))
    return dst


def main():
    image = cv2.imread("white_noise.jpg")
    res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    res = cv2.equalizeHist(res)
    res = cv2.Canny(res, 100, 200)
    circle_corners(res)
    distance = cv2.distanceTransform(res, cv2.DIST_L2, 3)
    filtered_img = median_filter(image, distance)

    cv2.imshow("Image", image)
    cv2.imshow("Median", filtered_img)
    cv2.imshow("Result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
