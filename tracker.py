import cv2
import numpy as np
import os


# 读取图片
img = cv2.imread("images/test.jpg")

if img is None:
    print("图片读取失败")
    exit()

# 复制用于绘制结果
result = img.copy()

# 转HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 蓝色范围（第一版）
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([140, 255, 255])

# 提取蓝色
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 去噪
kernel = np.ones((5, 5), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN, #开运算，先缩小小于5x5的白噪点，后膨胀至原大小用以消除噪点
    kernel
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE, #闭运算，同上原理，先膨胀黑，再缩小，补洞
    kernel
)

# 查找轮廓，不要第二个返回值
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("轮廓数量：", len(contours))

for contour in contours:

    area = cv2.contourArea(contour)

    if area < 50:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    cx = x + w // 2
    cy = y + h // 2

    print(
        f"找到蓝色区域 "
        f"面积={area:.1f} "
        f"中心=({cx},{cy})"
    )

    # 绿色框
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # 红色中心点
    cv2.circle(
        result,
        (cx, cy),
        6,
        (0, 0, 255),
        -1
    )

# 保存结果
os.makedirs("outputs", exist_ok=True)

ok1 = cv2.imwrite(
    "outputs/mask.jpg",
    mask
)

ok2 = cv2.imwrite(
    "outputs/result.jpg",
    result
)

print("mask 保存成功：", ok1)
print("result 保存成功：", ok2)