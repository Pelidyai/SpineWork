import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np

def show(win_name, img, scale=0.3):
    im = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                    interpolation=cv2.INTER_LINEAR)
    cv2.imshow(win_name, im)


def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * x + b))


def sigm_calc(vec, a, b):
    for j in range(len(vec)):
        vec[j] = sigmoid(j / len(vec), a, b)


def horizontal_gradient(in_image, space_window_width, a=15, b=10):
    img = in_image.copy()
    for i in range(img.shape[0]):
        vec1 = np.zeros((img.shape[1] - space_window_width) // 2)
        sigm_calc(vec1, a, b)
        vec3 = vec1[::-1]
        must_have = img.shape[1] - (img.shape[1] + space_window_width) // 2
        if len(vec1) < must_have:
            space_window_width += must_have - len(vec1)
        vec2 = ([1] * space_window_width)
        vec2 = np.array(vec2)
        vec2 = np.concatenate((vec2, vec3))
        vec1 = np.concatenate((vec1, vec2))
        line_copy = np.float64(img[i])
        line_copy *= vec1
        img[i] = np.uint8(line_copy)
    return img


def get_hist(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf = (cdf - cdf[0]) * 255 / (cdf[-1] - 1)
    cdf = cdf.astype(np.uint8)
    img2 = cdf[img]
    return img2


def contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(5, 5))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))

    lab = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return lab


def square_gradient(img, clear_percent=0.5):
    im = img.copy()
    im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    im = horizontal_gradient(im, int(im.shape[1] * clear_percent), 5, 0)
    im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im = horizontal_gradient(im, int(im.shape[1] * clear_percent), 5, 0)
    return im


def get_left_up_point(img, rec):
    flag = False
    points = None
    sec_points = None
    for i in range(rec[1], img.shape[0]):
        for j in range(rec[0], img.shape[1]):
            if (img[i][j] == (255, 255, 255)).all():
                points = (j, i)
                flag = True
                break
        if flag:
            break
    flag = False
    for i in range(rec[0], img.shape[1]):
        for j in range(rec[1], img.shape[0]):
            if (img[j][i] == (255, 255, 255)).all():
                sec_points = (i, j)
                flag = True
                break
        if flag:
            break
    stright_point = (sec_points[0], points[1])
    mid_point = (int((sec_points[0] + points[0]) / 2), int((sec_points[1] + points[1]) / 2))
    cur_point = int((stright_point[0] + mid_point[0]) / 2), int((stright_point[1] + mid_point[1]) / 2)
    a_tan = abs(np.arctan(((sec_points[0] - points[0]) / (sec_points[1] - points[1]))))
    if a_tan > 0.65:
        distance = abs(stright_point[1] - sec_points[1])
        main_point = (sec_points[0], int(stright_point[1] + distance * ((a_tan - 0.65) / 1.6)))
    else:
        distance = abs(stright_point[0] - points[0])
        main_point = (int(stright_point[0] + distance * (a_tan / 0.65)), points[1])
    cv2.drawMarker(img, points, (255, 0, 0), cv2.MARKER_SQUARE, 10, 2)
    cv2.drawMarker(img, sec_points, (0, 0, 255), cv2.MARKER_SQUARE, 10, 2)
    cur_point = int((cur_point[0] + main_point[0]) / 2), int((cur_point[1] + main_point[1]) / 2)
    return cur_point


def get_left_down_point(img, rec):
    flag = False
    points = None
    sec_points = None
    for i in range(rec[3] - 1, rec[1], -1):
        for j in range(rec[0], img.shape[1]):
            if (img[i][j] == (255, 255, 255)).all():
                points = (j, i)
                flag = True
                break
        if flag:
            break
    flag = False
    for i in range(rec[0], img.shape[1]):
        for j in range(rec[3] - 1, rec[1], -1):
            if (img[j][i] == (255, 255, 255)).all():
                sec_points = (i, j)
                flag = True
                break
        if flag:
            break
    stright_point = (sec_points[0], points[1])
    mid_point = (int((sec_points[0] + points[0]) / 2), int((sec_points[1] + points[1]) / 2))
    cur_point = int((stright_point[0] + mid_point[0]) / 2), int((stright_point[1] + mid_point[1]) / 2)
    a_tan = abs(np.arctan(((sec_points[0] - points[0]) / (sec_points[1] - points[1]))))
    if a_tan > 0.65:
        distance = abs(stright_point[1] - sec_points[1])
        main_point = (sec_points[0], int(stright_point[1] - distance * ((a_tan - 0.65) / 1.5)))
    else:
        distance = abs(stright_point[0] - points[0])
        main_point = (int(stright_point[0] + distance * (a_tan / 0.65)), points[1])
    cur_point = int((cur_point[0] + main_point[0]) / 2), int((cur_point[1] + main_point[1]) / 2)
    return cur_point


def get_right_up_point(img, rec):
    flag = False
    points = None
    sec_points = None
    for i in range(rec[0], img.shape[0]):
        for j in range(rec[2] - 1, rec[0], -1):
            if (img[i][j] == (255, 255, 255)).all():
                points = (j, i)
                flag = True
                break
        if flag:
            break
    flag = False
    for i in range(rec[2] - 1, rec[0], -1):
        for j in range(rec[0], img.shape[0]):
            if (img[j][i] == (255, 255, 255)).all():
                sec_points = (i, j)
                flag = True
                break
        if flag:
            break
    stright_point = (sec_points[0], points[1])
    mid_point = (int((sec_points[0] + points[0]) / 2), int((sec_points[1] + points[1]) / 2))
    cur_point = int((stright_point[0] + mid_point[0]) / 2), int((stright_point[1] + mid_point[1]) / 2)
    a_tan = abs(np.arctan(((sec_points[0] - points[0]) / (sec_points[1] - points[1]))))
    if a_tan > 0.65:
        distance = abs(stright_point[1] - sec_points[1])
        main_point = (sec_points[0], int(stright_point[1] + distance * (1 - (a_tan - 0.65) / 1.5)))
    else:
        distance = abs(stright_point[0] - points[0])
        main_point = (int(stright_point[0] - distance * (a_tan / 0.65)), points[1])
    cur_point = int((cur_point[0] + main_point[0]) / 2), int((cur_point[1] + main_point[1]) / 2)
    return cur_point


def get_right_down_point(img, rec):
    flag = False
    points = None
    sec_points = None
    for i in range(rec[3] - 1, rec[1], -1):
        for j in range(rec[2] - 1, rec[0], -1):
            if (img[i][j] == (255, 255, 255)).all():
                points = (j, i)
                flag = True
                break
        if flag:
            break
    flag = False
    for i in range(rec[2] - 1, rec[0], -1):
        for j in range(rec[3] - 1, rec[1], -1):
            if (img[j][i] == (255, 255, 255)).all():
                sec_points = (i, j)
                flag = True
                break
        if flag:
            break
    stright_point = (sec_points[0], points[1])
    mid_point = (int((sec_points[0] + points[0]) / 2), int((sec_points[1] + points[1]) / 2))
    cur_point = int((stright_point[0] + mid_point[0]) / 2), int((stright_point[1] + mid_point[1]) / 2)
    a_tan = abs(np.arctan(((sec_points[0] - points[0]) / (sec_points[1] - points[1]))))
    if a_tan > 0.65:
        distance = abs(stright_point[1] - sec_points[1])
        main_point = (sec_points[0], int(stright_point[1] - distance * (1 - (a_tan - 0.65) / 1.5)))
    else:
        distance = abs(stright_point[0] - points[0])
        main_point = (int(stright_point[0] - distance * (a_tan / 0.65)), points[1])
    cur_point = int((cur_point[0] + main_point[0]) / 2), int((cur_point[1] + main_point[1]) / 2)
    return cur_point


def get_math_points(img, rec):
    points = [get_left_down_point(img, rec),
              get_left_up_point(img, rec),
              get_right_up_point(img, rec),
              get_right_down_point(img, rec)]
    return points


def get_range(img, low, up):
    lower = np.array([low])
    upper = np.array([up])
    return cv2.inRange(img, lower, upper)


def point_relative_to_line(point, line) -> int:
    dx1 = line[1][0] - line[0][0]
    dy1 = line[1][1] - line[0][1]
    dx2 = line[0][0] - point[0]
    dy2 = line[0][1] - point[1]
    return dx1 * dy2 - dy1 * dx2


def is_point_between_lines(point, left_line, right_line) -> bool:
    relative_to_left_line = point_relative_to_line(point, left_line)
    relative_to_right_line = point_relative_to_line(point, right_line)
    between = relative_to_left_line * relative_to_right_line
    return between <= 0


def clear_mask_n_contours(mask, contour, first_line, second_line):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if (mask[i][j] == (255, 255, 255)).all() and is_point_between_lines((j, i), first_line, second_line):
                mask[i][j] = 0
            if (contour[i][j] == (255, 255, 255)).all() and is_point_between_lines((j, i), first_line, second_line):
                contour[i][j] = 0


def get_up_line(mask, contour, rec):
    flag = False
    point = None
    for i in range(rec[1], rec[3]):
        for j in range(rec[0], rec[2]):
            if (mask[i][j] == (255, 255, 255)).all():
                point = (j, i)
                flag = True
                break
        if flag:
            break
    line_points = []
    start_x = point[0]
    if start_x > mask.shape[1] / 2:
        start_x = int(point[0] + mask.shape[1] * 0.03)
        for j in range(start_x, int(rec[0] * 1.05), -1):
            for i in range(int(point[1] * 0.95), int(point[1] + mask.shape[0] * 0.2)):
                if contour[i][j] == 255:
                    line_points.append([j, i])
    else:
        start_x = int(point[0] - mask.shape[1] * 0.03)
        for j in range(start_x, int(rec[2] * 0.95)):
            for i in range(int(point[1] * 0.95), int(point[1] + mask.shape[0] * 0.2)):
                if contour[i][j] == 255:
                    line_points.append([j, i])
    line_points = np.array(line_points)
    vx, vy, cx, cy = cv2.fitLine(line_points, cv2.DIST_L2, 0, 0.01, 0.01)
    w = mask.shape[1]
    out_line = [(int(cx - vx * w), int(cy - vy * w) - 4), (int(cx + vx * w), int(cy + vy * w) - 4)]
    return out_line


def get_down_line(mask, contour, rec):
    flag = False
    point = None
    for i in range(rec[3], rec[1], -1):
        for j in range(rec[0], rec[2]):
            if (mask[i][j] == (255, 255, 255)).all():
                point = (j, i)
                flag = True
                break
        if flag:
            break
    line_points = []
    start_x = point[0]
    up = int(point[1] + mask.shape[0] * 0.05)
    if up >= mask.shape[0]:
        up = mask.shape[0] - 1
    down = int(point[1] * 0.8)
    if start_x > mask.shape[1] / 2:
        start_x = int(point[0] + mask.shape[1] * 0.03)
        for j in range(start_x, int(rec[0] * 1.05), -1):
            for i in range(down, up):
                if contour[i][j] == 255:
                    line_points.append([j, i])
    else:
        start_x = int(point[0] - mask.shape[1] * 0.03)
        for j in range(start_x, int(rec[2] * 0.95)):
            for i in range(down, up):
                if contour[i][j] == 255:
                    line_points.append([j, i])
    line_points = np.array(line_points)
    vx, vy, cx, cy = cv2.fitLine(line_points, cv2.DIST_L2, 0, 0.01, 0.01)
    w = mask.shape[1]
    out_line = [(int(cx - vx * w), int(cy - vy * w) + 4), (int(cx + vx * w), int(cy + vy * w) + 4)]
    return out_line


def get_left_line(mask, contour, rec):
    flag = False
    point = None
    for j in range(rec[0], rec[2]):
        for i in range(rec[1], rec[3]):
            if (mask[i][j] == (255, 255, 255)).all():
                point = (j, i)
                flag = True
                break
        if flag:
            break
    line_points = []
    start_y = point[1]
    if start_y > mask.shape[0] / 2:
        start_y = int(point[1] + mask.shape[0] * 0.03)
        for j in range(int(point[0] * 0.95), int(point[0] + mask.shape[1] * 0.15)):
            for i in range(start_y, int(rec[1] * 1.05), -1):
                if contour[i][j] == 255:
                    line_points.append([j, i])
    else:
        start_y = int(point[1] - mask.shape[0] * 0.03)
        for j in range(int(point[0] * 0.95), int(point[0] + mask.shape[1] * 0.15)):
            for i in range(start_y, int(rec[3] * 0.95)):
                if contour[i][j] == 255:
                    line_points.append([j, i])
    line_points = np.array(line_points)
    vx, vy, cx, cy = cv2.fitLine(line_points, cv2.DIST_L2, 0, 0.01, 0.01)
    w = mask.shape[1]
    out_line = [(int(cx - vx * w) - 3, int(cy - vy * w)), (int(cx + vx * w) - 3, int(cy + vy * w))]
    return out_line


def get_right_line(mask, contour, rec):
    flag = False
    point = None
    for j in range(rec[2], rec[0], -1):
        for i in range(rec[1], rec[3]):
            if (mask[i][j] == (255, 255, 255)).all():
                point = (j, i)
                flag = True
                break
        if flag:
            break
    line_points = []
    start_y = point[1]
    right = int(point[0] + mask.shape[1] * 0)
    if right >= mask.shape[1]:
        right = mask.shape[1] - 1
    left = int(point[0] * 0.85)
    if start_y > mask.shape[0] / 2 - 5:
        start_y = int(point[1] + mask.shape[0] * 0)
        for j in range(left, right):
            for i in range(start_y, int(rec[1] * 1.05), -1):
                if contour[i][j] == 255:
                    line_points.append([j, i])
    else:
        start_y = int(point[1] - mask.shape[0] * 0.05)
        for j in range(left, right):
            for i in range(start_y, int(rec[3] * 0.95)):
                if contour[i][j] == 255:
                    line_points.append([j, i])
    line_points = np.array(line_points)
    vx, vy, cx, cy = cv2.fitLine(line_points, cv2.DIST_L2, 0, 0.01, 0.01)
    w = mask.shape[1]
    out_line = [(int(cx - vx * w) + 3, int(cy - vy * w)), (int(cx + vx * w) + 3, int(cy + vy * w))]
    return out_line


def get_line(p1, p2):
    a = (p1[1] - p2[1])
    b = (p2[0] - p1[0])
    c = (p1[0] * p2[1] - p2[0] * p1[1])
    return a, b, -c


def intersection(L1, L2):
    d = L1[0] * L2[1] - L1[1] * L2[0]
    dx = L1[2] * L2[1] - L1[1] * L2[2]
    dy = L1[0] * L2[2] - L1[2] * L2[0]
    if d != 0:
        x = dx / d
        y = dy / d
        point = (int(x), int(y))
        return point
    else:
        return False


def get_cross_point(f_line, s_line):
    l1 = get_line(f_line[0], f_line[1])
    l2 = get_line(s_line[0], s_line[1])
    return intersection(l1, l2)


def get_point(f_line, s_line, frontiers, mask):
    point = get_cross_point(f_line, s_line)
    if point and point[0] >= 0 and point[1] >= 0 and point[0] <= mask.shape[1] + 1 and point[1] <= mask.shape[0] + 1:
        # cv2.drawMarker(vertebra, point, (255, 255, 0), cv2.MARKER_STAR, 10, 1)
        return point
    else:
        left = get_cross_point(s_line, frontiers[1])
        if not left or not left[0] >= 0 or not left[1] >= 0 \
                or not left[0] <= mask.shape[1] + 1 or not left[1] <= mask.shape[0] + 1:
            left = get_cross_point(s_line, frontiers[0])
        down = get_cross_point(f_line, frontiers[0])
        if not down or not down[0] >= 0 or not down[1] >= 0 \
                or not down[0] <= mask.shape[1] + 1 or not down[1] <= mask.shape[0] + 1:
            down = get_cross_point(f_line, frontiers[1])
        point = (int((down[0] + left[0]) / 2), int((down[1] + left[1]) / 2))
    return point


def get_line_points(mask, contour, rec, small_recs_count):
    frontier = ([(0, 0), (mask.shape[1] - 1, 0)],  # up
                [(0, mask.shape[0] - 1), (mask.shape[1] - 1, mask.shape[0] - 1)],  # down
                [(0, mask.shape[0] - 1), (0, 0)],  # left
                [(mask.shape[1] - 1, mask.shape[0] - 1), (mask.shape[1] - 1, 0)],  # right
                )
    lines = [get_up_line(mask, contour, rec),
             get_down_line(mask, contour, rec),
             get_left_line(mask, contour, rec),
             get_right_line(mask, contour, rec)]
    # for line in lines:
    #     cv2.line(vertebra, line[0], line[1], (0, 0, 255), 1)
    if small_recs_count > 0:
        for _ in range(5):
            clear_mask_n_contours(mask, contour, frontier[0], lines[0])
            clear_mask_n_contours(mask, contour, frontier[1], lines[1])
            lines[0] = get_up_line(mask, contour, rec)
            lines[1] = get_down_line(mask, contour, rec)
            # for line in lines:
            #     cv2.line(vertebra, line[0], line[1], (0, 0, 255), 1)
    # for line in lines:
    #     cv2.line(vertebra, line[0], line[1], (255, 0, 0), 1)
    points = [get_point(lines[1], lines[2], [frontier[1], frontier[2]], mask),
              get_point(lines[0], lines[2], [frontier[0], frontier[2]], mask),
              get_point(lines[0], lines[3], [frontier[0], frontier[3]], mask),
              get_point(lines[1], lines[3], [frontier[1], frontier[3]], mask)]
    return points, contour


def get_points(mask, contour, rec, small_recs_count):
    global vertebra
    line_points, contour = get_line_points(mask, contour, rec, small_recs_count)
    math_points = get_math_points(mask, rec)

    ret_points = []
    for point in math_points:
        vertebra = cv2.drawMarker(vertebra, point, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)
    for point in line_points:
        vertebra = cv2.drawMarker(vertebra, point, (255, 0, 0), cv2.MARKER_STAR, 10, 1)
    for i in range(len(math_points)):
        ret_points.append((int((math_points[i][0]+line_points[i][0])/2), int((math_points[i][1]+line_points[i][1])/2)))
    for point in ret_points:
        vertebra = cv2.drawMarker(vertebra, point, (0, 255, 255), cv2.MARKER_DIAMOND, 5, 2)
    cv2.line(vertebra, ret_points[0], ret_points[1], (0, 255, 255), 1)
    cv2.line(vertebra, ret_points[1], ret_points[2], (0, 255, 255), 1)
    cv2.line(vertebra, ret_points[2], ret_points[3], (0, 255, 255), 1)
    cv2.line(vertebra, ret_points[3], ret_points[0], (0, 255, 255), 1)
    # show('res', vertebra, 1.3)
    return ret_points, contour


def get_vecs_from_contour(contour):
    xs = []
    ys = []
    for i in range(0, contour.shape[0], 1):
        for j in range(0, contour.shape[1], 1):
            if (contour[i][j] == (255, 255, 255)).all():
                xs.append(j)
                ys.append(i)
    return xs, ys


def get_3d_points(s_projection, a_projection):
    x, y = get_vecs_from_contour(s_projection)
    xy, z = get_vecs_from_contour(a_projection)
    y = y[::-1]
    z = z[::-1]
    # xy = xy[::-1]
    xs = []
    ys = []
    zs = []
    delta = 1
    for j in range(0, len(y), delta):
        for i in range(0, len(x), delta):
            # for k in range(0, len(z), delta):
            if z[i] == y[j]:
                xs.extend([x[i]])
                zs.extend([z[i]])
                ys.extend([xy[j]])

    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(xs, ys, zs, label='vertebra', marker='.', s=2)
    ax.legend()
    plt.show()


def get_opening(vertebra):
    dx = int(vertebra.shape[1] * 0.04)
    vertebra = vertebra[dx:vertebra.shape[0] - dx, dx:vertebra.shape[1] - dx]
    vertebra = cv2.cvtColor(vertebra, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(vertebra, (1, 1))
    # show('blur', blur, 1.5)
    hist = square_gradient(blur, 0.3)
    # show('grad', hist, 1.5)
    hist = get_hist(hist)
    # show('hist', hist, 1.5)
    hist_range = get_range(hist, 140, 255)
    # show('slice', hist_range, 1.5)

    # show('hist_range', hist_range, 1.3)

    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(hist_range, kernel, iterations=1)
    # show('erosion', erosion, 1.5)
    opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
    return opening


if __name__ == '__main__':
    files = os.listdir('verts')
    for file in files:
        vertebra = cv2.imread(os.path.join('verts', file))
        opening = get_opening(vertebra)  # todo use this in the future
        show('opening', opening, 1.5)
        cv2.waitKey(0)
        # cv2.imwrite(os.path.join('vert_masks', file), opening)
        continue
        # show('opening', opening, 1.3)

        contours, hierarchy = cv2.findContours(
            opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        opening = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
        recs = []
        cont_img = np.zeros((opening.shape[0], opening.shape[1]), np.uint8)
        small_recs_count = 0
        if len(contours) != 0:
            for (j, c) in enumerate(contours):
                area = cv2.contourArea(c)
                if area > 400:
                    r = cv2.boundingRect(c)
                    # cv2.rectangle(opening, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, i * 100, 255), 2)
                    recs.append(r)
                if area > 20:
                    small_recs_count += 1
                    r = cv2.boundingRect(c)
                    # cv2.rectangle(opening, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), (0, i * 100, 255), 2)
                cv2.drawContours(cont_img, c, -1, (255, 255, 255), 1)
                # print(c)
        small_recs_count -= len(recs)
        rec = [recs[0][0], recs[0][1], recs[0][0] + recs[0][2], recs[0][1] + recs[0][3]]
        if len(recs) == 2:
            rec[0] = recs[1][0]
            rec[1] = recs[1][1]
        # cv2.rectangle(opening, (rec[0], rec[1]), (rec[2], rec[3]), (250,0, 255), 2)
        # show('contours', cont_img, 1.3)
        # clean_contour(cont_img, rec)
        vertebra = cv2.cvtColor(vertebra, cv2.COLOR_GRAY2BGR)
        # points, cont_img = get_points(opening, cont_img, rec, small_recs_count)
        # show('contours', cont_img, 1.3)
        # get_3d_points(cont_img, cont_img)

        cv2.waitKey(0)
        break

    cv2.waitKey(0)
