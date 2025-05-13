import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv

# ==== 설정 ====
MAP_IMAGE_PATH = 'Track.png'
CSV_PATH = 'global_path.csv'
RESOLUTION = 0.00212
ORIGIN = [0.0, 0.0, 0.0]  # 좌측 하단 기준 (meter)

## 생성된 경로를 시각화해주는 코드

# ==== CSV 로드 ====
def load_path_from_csv(filename):
    coords = []
    with open(filename, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x, y, dotted = float(row["x"]), float(row["y"]), int(row["dotted"])
            coords.append((x, y, dotted))
    return coords


# ==== 시각화 ====
def visualize_path_on_image(image_path, coords):
    img = mpimg.imread(image_path)
    height = img.shape[0]

    # meter → pixel 변환
    def to_pixel(x, y):
        px = (x - ORIGIN[0]) / RESOLUTION
        py = height - ((y - ORIGIN[1]) / RESOLUTION)
        return px, py

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img)

    prev = None
    for i in range(len(coords)):
        x, y, dotted = coords[i]
        px, py = to_pixel(x, y)

        if prev:
            x_prev, y_prev = to_pixel(prev[0], prev[1])
            color = 'red' if dotted else 'green'
            ax.plot([x_prev, px], [y_prev, py], color=color, linewidth=2)

        prev = (x, y)

    ax.set_title("Global Path Visualization with Dotted Feature")
    plt.show()


if __name__ == '__main__':
    coords = load_path_from_csv(CSV_PATH)
    visualize_path_on_image(MAP_IMAGE_PATH, coords)
