import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.image as mpimg
import numpy as np
import csv

## GUI를 통해 waypoint를 입력 받고 그 waypoint를 이용해 경로 계산 

# ==== 설정 ====
MAP_IMAGE_PATH = 'Track.png'
GRAPHML_FILE = 'Competition_track_graph.graphml'
RESOLUTION = 0.00212
ORIGIN = [0.0, 0.0, 0.0]


# ==== Hermite 보간 + dotted 처리 ====
def hermite_interpolate_with_dotted(points, dotted_flags, per_segment=5):
    interpolated = []

    def compute_tangent(p_prev, p_next):
        dx = (p_next[0] - p_prev[0]) * 0.5
        dy = (p_next[1] - p_prev[1]) * 0.5
        return (dx, dy)

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        dotted = dotted_flags[i]

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
            interpolated.append((x, y, dotted))

    interpolated.append((*points[-1], dotted_flags[-1]))
    return interpolated


# ==== CSV 저장 ====
def save_path_to_csv_with_dotted(path_coords, filename="global_path.csv"):
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "dotted"])
        for x, y, dotted in path_coords:
            writer.writerow([x, y, int(dotted)])


# ==== GraphML 로드 + 픽셀 변환 + 원본 좌표 + dotted 읽기 ====
def load_graph_and_waypoints(graphml_path, img_height):
    G = nx.read_graphml(graphml_path)
    positions = {}
    waypoint_dict = {}

    for node_id, data in G.nodes(data=True):
        node_int_id = int(node_id)
        x = float(data['x'])
        y = float(data['y'])
        waypoint_dict[node_int_id] = (x, y)

        px = (x - ORIGIN[0]) / RESOLUTION
        py = img_height - ((y - ORIGIN[1]) / RESOLUTION)
        positions[node_int_id] = (px, py)

    # Edge dotted 정보 정규화 (문자열 → bool로)
    for _, _, attrs in G.edges(data=True):
        dotted_val = attrs.get("dotted", "false")
        if isinstance(dotted_val, str):
            attrs["dotted"] = dotted_val.lower() == "true"
        else:
            attrs["dotted"] = bool(dotted_val)


    return G, positions, waypoint_dict


# ==== 클릭 GUI ====
class WaypointSelector:
    def __init__(self, graph, pos, img, waypoint_dict):
        self.G = graph
        self.pos = pos
        self.img = img
        self.waypoint_dict = waypoint_dict
        self.selected = []
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.path_lines = []  # 경로 라인을 저장할 리스트

    def draw(self):
        self.ax.imshow(self.img)
        # 노드 표시
        for node, (x, y) in self.pos.items():
            self.ax.plot(x, y, 'ro', markersize=3)
        self.ax.set_title('Click to select waypoints (close window to finish)')
        plt.show()

    def draw_path(self, start_node, end_node):
        try:
            # 실제 트랙을 따라가는 경로 찾기
            path = nx.shortest_path(self.G, source=str(start_node), target=str(end_node), weight='length')
            
            # 경로상의 모든 노드의 좌표를 수집
            path_coords = []
            for node in path:
                node_int = int(node)
                if node_int in self.waypoint_dict:
                    x, y = self.waypoint_dict[node_int]
                    px = (x - ORIGIN[0]) / RESOLUTION
                    py = self.img.shape[0] - ((y - ORIGIN[1]) / RESOLUTION)
                    path_coords.append((px, py))
            
            # 경로 그리기
            if len(path_coords) > 1:
                x_coords = [coord[0] for coord in path_coords]
                y_coords = [coord[1] for coord in path_coords]
                line, = self.ax.plot(x_coords, y_coords, 'g-', linewidth=2)
                self.path_lines.append(line)
                self.fig.canvas.draw()
                
        except nx.NetworkXNoPath:
            print(f"Warning: No path found between {start_node} and {end_node}")

    def onclick(self, event):
        if event.inaxes:
            click_x, click_y = event.xdata, event.ydata
            closest_node = min(
                self.pos.items(),
                key=lambda item: (item[1][0] - click_x) ** 2 + (item[1][1] - click_y) ** 2
            )[0]

            if len(self.selected) > 0:
                prev_node = self.selected[-1]
                # 이전 경로 라인 제거
                for line in self.path_lines:
                    line.remove()
                self.path_lines.clear()
                
                # 새로운 경로 그리기
                self.draw_path(prev_node, closest_node)

            self.selected.append(closest_node)
            print(f"Selected: {closest_node}")
            self.ax.plot(self.pos[closest_node][0], self.pos[closest_node][1], 'go')
            self.fig.canvas.draw()

    def get_selected_waypoints(self):
        return self.selected


# ==== 실행 ====
def main():
    img = mpimg.imread(MAP_IMAGE_PATH)
    img_height = img.shape[0]

    G, positions, waypoint_dict = load_graph_and_waypoints(GRAPHML_FILE, img_height)

    selector = WaypointSelector(G, positions, img, waypoint_dict)
    selector.draw()

    selected_ids = selector.get_selected_waypoints()
    path_coords = []
    dotted_flags = []

    for i in range(len(selected_ids) - 1):
        u = str(selected_ids[i])
        v = str(selected_ids[i + 1])
        try:
            # 트랙을 따라가는 경로 찾기
            path = nx.shortest_path(G, source=u, target=v, weight='length')
            
            # 경로상의 모든 노드의 좌표를 수집
            segment_coords = []
            for node in path:
                node_int = int(node)
                if node_int in waypoint_dict:
                    segment_coords.append(waypoint_dict[node_int])
            
            # 경로상의 모든 엣지 dotted 여부 확인
            all_dotted = True
            for j in range(len(path) - 1):
                a, b = path[j], path[j + 1]
                if G.has_edge(a, b):
                    if not G[a][b].get("dotted", False):
                        all_dotted = False
                        break
                elif G.has_edge(b, a):
                    if not G[b][a].get("dotted", False):
                        all_dotted = False
                        break
                else:
                    all_dotted = False
                    break
            
            # 각 좌표에 대해 dotted 플래그 추가
            for coord in segment_coords:
                path_coords.append(coord)
                dotted_flags.append(all_dotted)
            
        except nx.NetworkXNoPath:
            print(f"Warning: No path found between {u} and {v}")
            # 연결 실패 시 이전 점과 현재 점을 직선으로 연결
            if i > 0:
                path_coords.append(waypoint_dict[selected_ids[i]])
                dotted_flags.append(False)
            path_coords.append(waypoint_dict[selected_ids[i + 1]])
            dotted_flags.append(False)

    # 마지막 점이 아직 추가되지 않았다면 추가
    if selected_ids and waypoint_dict[selected_ids[-1]] not in path_coords:
        path_coords.append(waypoint_dict[selected_ids[-1]])
        dotted_flags.append(dotted_flags[-1] if dotted_flags else False)

    # 경로 보간
    interpolated_coords = hermite_interpolate_with_dotted(path_coords, dotted_flags, per_segment=5)
    save_path_to_csv_with_dotted(interpolated_coords, "global_path.csv")

    print("\n✅ 경로가 global_path.csv에 저장되었습니다.")


if __name__ == "__main__":
    main()
