import cv2


class BedSelector:

    def __init__(self, image_path):

        # 读取图片
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise FileNotFoundError(image_path)

        # 原图尺寸
        self.height, self.width = self.image.shape[:2]

        # 当前显示图
        self.display = self.image.copy()

        # 保存四个点
        self.points = []

        # 窗口名字
        self.window_name = "Bed Selector"

    def create_window(self):

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            self.window_name,
            600,
            800
        )

        cv2.setMouseCallback(
            self.window_name,
            self.mouse_callback
        )


    def mouse_callback(self, event, x, y, flags, param):

    # 左键添加点
        if event == cv2.EVENT_LBUTTONDOWN:

            if len(self.points) >= 4:
                return

            self.points.append((x, y))

            print(f"第{len(self.points)}个点：({x},{y})")

            self.redraw()

        # 右键撤销
        elif event == cv2.EVENT_RBUTTONDOWN:

            if len(self.points) == 0:
                return

            removed = self.points.pop()

            print(f"删除点：{removed}")

            self.redraw()


    def redraw(self):

        self.display = self.image.copy()

        # 画点
        for i, point in enumerate(self.points):

            cv2.circle(
                self.display,
                point,
                8,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                self.display,
                str(i + 1),
                (point[0] + 10, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        # 连线
        if len(self.points) >= 2:

            for i in range(len(self.points) - 1):

                cv2.line(
                    self.display,
                    self.points[i],
                    self.points[i + 1],
                    (255, 0, 0),
                    2
                )

        # 闭合
        if len(self.points) == 4:

            cv2.line(
                self.display,
                self.points[3],
                self.points[0],
                (255, 0, 0),
                2
            )