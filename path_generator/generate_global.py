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
    waypoint_ids = ['263', '266', '268', '270', '242', '245', '248', '249', '250', '255', '256', '257', '54', '55', '309', '310', '313', '316', '331', '333', '374', '375', '376', '377', '378', '381', '382', '383', '384', '391', '392', '393', '394', '395', '397', '337', '338', '398', '486', '488', '491', '494', '497', '500', '425', '428', '434', '438', '439', '440', '441', '442', '84', '79', '93', '88', '102', '97', '124', '125', '126', '127', '128', '129', '75', '70', '185', '186', '207', '211', '212', '216', '40', '45', '37', '30', '25', '100', '103', '122', '125', '126', '127', '128', '129', '130', '75', '76', '72', '28', '31', '16', '17', '2', '9', '7', '152', '154', '156', '158', '160', '162', '164', '82', '85', '79', '182', '93', '94', '90', '14', '17', '2', '3', '221', '44', '202', '206', '187', '190', '191', '192', '239']  
    waypoint_ids = list(map(int, waypoint_ids)) 
    waypoint_dict = parse_graphml(GRAPHML_FILE)
    raw_coords = [waypoint_dict[i] for i in waypoint_ids]
    interpolated_coords = hermite_interpolate(raw_coords, per_segment=3)
    save_path_to_csv(interpolated_coords, "global_path.csv")

    print("경로가 global_path.csv에 저장되었습니다.")
