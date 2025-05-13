import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.image as mpimg
from PyQt5.QtWidgets import QApplication
import sys

# 맵 설정
MAP_IMAGE_PATH = 'Track.png'
RESOLUTION = 0.00212
ORIGIN = [0.0, 0.0, 0.0]

# 그래프 로드
def load_graph(graphml_path, img_height):
    G = nx.read_graphml(graphml_path)
    positions = {}

    # y 좌표만 반전해서 저장
    for node_id, data in G.nodes(data=True):
        x = float(data['x'])  # [m]
        y = float(data['y'])
        px = (x - ORIGIN[0]) / RESOLUTION
        py = img_height - ((y - ORIGIN[1]) / RESOLUTION)  # y 반전

        positions[node_id] = (px, py)
    return G, positions

# 마우스로 경로 클릭
class WaypointSelector:
    def __init__(self, graph, pos, img):
        self.G = graph
        self.pos = pos
        self.img = img
        self.selected = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6)) 
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def draw(self):
        self.ax.imshow(self.img)
        for node, (x, y) in self.pos.items():
            self.ax.plot(x, y, 'bo')
            # self.ax.text(x, y, str(node), fontsize=8, color='blue')
        self.ax.set_title('Click to select waypoints (close window to finish)')
        plt.show()

    def onclick(self, event):
        if event.inaxes:
            click_x, click_y = event.xdata, event.ydata
            # 가장 가까운 노드 찾기
            closest_node = min(
                self.pos.items(),
                key=lambda item: (item[1][0] - click_x) ** 2 + (item[1][1] - click_y) ** 2
            )[0]

            if len(self.selected) > 0:
                prev_node = self.selected[-1]
                # 두 노드 간 선 그리기
                x_values = [self.pos[prev_node][0], self.pos[closest_node][0]]
                y_values = [self.pos[prev_node][1], self.pos[closest_node][1]]
                self.ax.plot(x_values, y_values, 'g-', linewidth=2)

            self.selected.append(closest_node)
            print(f"Selected: {closest_node}")
            self.ax.plot(self.pos[closest_node][0], self.pos[closest_node][1], 'go')
            self.fig.canvas.draw()

    def get_selected_waypoints(self):
        return self.selected

def main():
    # app = QApplication(sys.argv)  # PyQt5 용도 (필요시)

    graph_path = 'Competition_track_graph.graphml'
    img = mpimg.imread(MAP_IMAGE_PATH)
    img_height = img.shape[0]
    G, positions = load_graph(graph_path, img_height)

    selector = WaypointSelector(G, positions, img)
    selector.draw()
    
    # 출력된 경로
    print("Final waypoint path:", selector.get_selected_waypoints())

if __name__ == "__main__":
    main()
