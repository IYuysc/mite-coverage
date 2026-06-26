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