import xml.etree.ElementTree as ET
import numpy as np
import csv

GRAPHML_FILE = "Competition_track_graph.graphml"   

def parse_graphml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
    nodes = root.findall(".//graphml:node", ns)

    waypoint_dict = {}
    for node in nodes:
        node_id = int(node.attrib['id'])
        x = float(node.find("graphml:data[@key='d0']", ns).text)
        y = float(node.find("graphml:data[@key='d1']", ns).text)
        waypoint_dict[node_id] = (x, y)

    return waypoint_dict

def hermite_interpolate(points, per_segment=5):
    interpolated = []

    def compute_tangent(p_prev, p_next):
        dx = (p_next[0] - p_prev[0]) * 0.5
        dy = (p_next[1] - p_prev[1]) * 0.5
        return (dx, dy)

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]

        # 양끝점의 tangent 계산
        t0 = compute_tangent(points[i - 1] if i > 0 else p0, p1)
        t1 = compute_tangent(p0, points[i + 2] if i + 2 < len(points) else p1)

        for j in range(per_segment):
            s = j / float(per_segment)

            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2

            x = h00 * p0[0] + h10 * t0[0] + h01 * p1[0] + h11 * t1[0]
            y = h00 * p0[1] + h10 * t0[1] + h01 * p1[1] + h11 * t1[1]

            interpolated.append((x, y))

    interpolated.append(points[-1])
    return interpolated


def save_path_to_csv(path_coords, filename="global_path.csv"):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for x, y in path_coords:
            writer.writerow([x, y])

if __name__ == '__main__':
    waypoint_ids = [263, 264, 265, 266, 267, 268, 269, 270, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252,
                    253, 254, 255, 256, 257, 54, 55, 305, 306, 309, 310, 311, 312, 313, 314, 315, 316, 317, 331, 332,
                    333, 368, 369, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 390, 391, 392, 393,
                    394, 395, 396, 397, 337, 338, 398, 399, 400, 483, 485, 487, 489, 491, 493, 495, 497, 499, 501, 425,
                    429, 434, 438, 439, 440, 441, 442, 443, 79, 181, 183, 93, 95, 88, 145, 102, 105, 97, 122, 123, 125,
                    126, 127, 128, 129, 130, 70, 186, 210, 211, 212, 213, 216, 40, 43, 219, 4, 7, 151, 152, 153, 154, 155, 
                    156, 157, 158, 159, 160, 161, 162, 163, 164, 82, 83, 406, 407, 408, 409, 410, 411, 422, 423, 463, 464, 466,
                    467, 468, 469, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 520, 402, 405, 340, 341, 318, 319,
                    321, 322, 323, 324, 325, 326, 56, 49, 273, 284, 289, 291, 201, 42, 45, 39, 202, 205, 207, 187, 189, 
                    190, 191, 192, 223, 225, 227, 229, 235, 239, 243]  

    waypoint_dict = parse_graphml(GRAPHML_FILE)
    raw_coords = [waypoint_dict[i] for i in waypoint_ids]
    interpolated_coords = hermite_interpolate(raw_coords, per_segment=3)
    save_path_to_csv(interpolated_coords, "global_path.csv")

    print("경로가 global_path.csv에 저장되었습니다.")
