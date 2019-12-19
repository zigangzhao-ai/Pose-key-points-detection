# coding=utf-8
# /usr/bin/env python

"""
Author: Rocsky
Description: 
"""
import os
import argparse
import cv2
import numpy as np
import random


def arctan(xy):
    return np.arctan(xy[1] / xy[0])


def cal_coord(point_src, center, angle, ratio):
    if isinstance(center, (list, tuple)):
        center = np.array(center)
    r = np.sqrt(np.power(point_src[0] - center, 2).sum())
    r = r * ratio
    ag = arctan(point_src[2] - center)
    ag_list = np.array([ag, np.pi - ag, ag + np.pi, np.pi * 2 - ag]) + angle / 180 * np.pi
    coords = []
    for a in ag_list:
        coords.append([r * np.cos(a), r * np.sin(a)])
    return np.array(coords) + center


# for i in range(4):
#     cv2.line(contour, tuple(point_dst[i % 4]), tuple(point_dst[(i + 1) % 4]), (255, 255, 255), 2)
def compose_back_fore(back, fore, difficult):
    thickness = 2
    fh, fw, _ = fore.shape
    rh = np.random.uniform(1.1, 1.5)
    rw = np.random.uniform(1.1, 1.5)
    bh = int(rh * fh + 0.5)
    bw = int(rw * fw + 0.5)
    back = cv2.resize(back, (bw, bh))
    lt = (int(bw / 3), int(bh / 3))
    rt = (int(bw - bw / 3), int(bh / 3))
    rb = (int(bw - bw / 3), int(bh - bh / 3))
    lb = (int(bw / 3), int(bh - bh / 3))
    warp = np.zeros_like(back)
    # Random Location
    x = np.random.randint(0, bw - fw)
    y = np.random.randint(0, bh - fh)
    warp[y:y + fh, x:x + fw] = fore
    # change
    mask = np.zeros((bh, bw), dtype=np.uint8)
    mask[y:y + fh, x:x + fw] = 1
    rand = np.random.randint(0, 3)
    if rand == 0:
        # Perspective
        if difficult:
            over = int(min(lt) / 4)
        else:
            over = 0
        point_src = np.array([[x, y], [fw + x, y], [fw + x, fh + y], [x, fh + y]], dtype=np.float32)
        point_dst = np.array([[np.random.randint(-over, lt[0]), np.random.randint(-over, lt[1])],
                              [np.random.randint(rt[0], bw + over), np.random.randint(-over, rt[1])],
                              [np.random.randint(rb[0], bw + over), np.random.randint(rb[1], bh + over)],
                              [np.random.randint(-over, lb[0]), np.random.randint(lb[1], bh + over)]],
                             dtype=np.float32)
        mat = cv2.getPerspectiveTransform(point_src, point_dst)
        warp = cv2.warpPerspective(warp, mat, (bw, bh))
        mask = cv2.warpPerspective(mask, mat, (bw, bh))
        # change
        major = cv2.__version__.split('.')[0]
        if major == '3':
            _, con, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            con, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        assert len(con) == 1
        # change
        contour = np.zeros((bh, bw), dtype=np.uint8)
        con_mask = np.zeros((bh, bw), dtype=np.uint8)
        contour = cv2.drawContours(contour, con, 0, 1, thickness)
        con_mask = cv2.drawContours(con_mask, con, 0, 1, 5)
        compose = cv2.bitwise_or(warp, cv2.bitwise_and(back, 255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * 255))
    elif rand == 1:
        # Rotation
        point_src = np.array([[x, y], [fw + x, y], [fw + x, fh + y], [x, fh + y]], dtype=np.float32)
        if difficult:
            center = tuple(point_src.mean(axis=0))
            angle = np.random.uniform(0, 360)
            ratio = 1
        else:
            center = tuple(point_src.mean(axis=0))
            r = np.sqrt(np.power(point_src[0] - np.array(center), 2).sum())
            dis = [center[0], center[1], bw - center[0], bh - center[1]]
            angle_list = []
            ag = arctan(point_src[2] - np.array(center))
            for i, d in enumerate(dis):
                if d >= r:
                    max_ag = 1
                else:
                    max_ag = np.arcsin(d / r)
                if i % 2 == 0:
                    angle_list.append(max_ag + ag - np.pi / 2)
                else:
                    angle_list.append(max_ag - ag)

            min_angle = min(angle_list) / np.pi * 180
            angle = np.random.uniform(-min_angle, min_angle)
            ratio = 1

        point_dst = cal_coord(point_src, center, angle, ratio)
        mat = cv2.getRotationMatrix2D(center, angle, ratio)
        warp = cv2.warpAffine(warp, mat, (bw, bh))
        mask = cv2.warpAffine(mask, mat, (bw, bh))
        # change
        # con, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        major = cv2.__version__.split('.')[0]
        if major == '3':
            _, con, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            con, _ = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        assert len(con) == 1
        # change
        contour = np.zeros((bh, bw), dtype=np.uint8)
        con_mask = np.zeros((bh, bw), dtype=np.uint8)
        contour = cv2.drawContours(contour, con, 0, 1, thickness)
        con_mask = cv2.drawContours(con_mask, con, 0, 1, 5)
        compose = cv2.bitwise_or(warp, cv2.bitwise_and(back, 255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * 255))
    elif rand == 2:
        # Shift
        if difficult:
            rx = np.random.randint(int(-bw / 10), int(bw / 10 * 3))
            ry = np.random.randint(int(-bh / 10), int(bh / 10 * 3))
        else:
            rx = 0
            ry = 0
        point_src = np.array([[x, y], [fw + x, y], [fw + x, fh + y], [x, fh + y]], dtype=np.float32)
        point_dst = point_src + np.array([rx, ry])
        mat = np.array([[1, 0, rx], [0, 1, ry]], dtype=np.float32)
        warp = cv2.warpAffine(warp, mat, (bw, bh))
        mask = cv2.warpAffine(mask, mat, (bw, bh))
        # change
        # con, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        major = cv2.__version__.split('.')[0]
        if major == '3':
            _, con, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            con, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        assert len(con) == 1
        # change
        contour = np.zeros((bh, bw), dtype=np.uint8)
        con_mask = np.zeros((bh, bw), dtype=np.uint8)
        contour = cv2.drawContours(contour, con, 0, 1, thickness)
        con_mask = cv2.drawContours(con_mask, con, 0, 1, 5)
        compose = cv2.bitwise_or(warp, cv2.bitwise_and(back, 255 - cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) * 255))

    return compose, mask, contour * 255, point_dst, con_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='输入图片文件夹地址 以及 保存图片的地址')
    parser.add_argument('--foreground_root', type=str, default='sources/foreground')
    parser.add_argument('--background_root', type=str, default='sources/background')
    parser.add_argument('--images_root', type=str, default='datasets/images')
    parser.add_argument('--annotations_root', type=str, default='datasets/annotations')
    args = parser.parse_args()
    foreground_root = args.foreground_root
    background_root = args.background_root
    images_root = args.images_root
    annotations_root = args.annotations_root
    foreground_list = os.listdir(foreground_root)
    background_list = os.listdir(background_root)
    i = 1
    lines = []
    while True:
        fore = cv2.imread(os.path.join(foreground_root, random.choice(foreground_list)))
        back = cv2.imread(os.path.join(background_root, random.choice(background_list)))
        compose, mask, contour, point, con_mask = compose_back_fore(back, fore, False)
        blur = cv2.GaussianBlur(compose, (7, 7), sigmaX=0)
        # blur = cv2.bitwise_and(blur, con_mask)
        compose[con_mask > 0] = blur[con_mask > 0]
        cv2.imwrite(os.path.join(images_root, 'image_%05d.png' % i), compose)
        cv2.imwrite(os.path.join(annotations_root, 'mask', 'image_%05d.png' % i), mask)
        cv2.imwrite(os.path.join(annotations_root, 'contour', 'image_%05d.png' % i), contour)
        np.savetxt(os.path.join(annotations_root, 'point', 'image_%05d.txt' % i), point, fmt='%.3f')
        min_x, min_y = point.min(axis=0)
        max_x, max_y = point.max(axis=0)
        h, w, _ = compose.shape
        difficult = np.any(point < 0) or np.any(point[:, 0] >= w) or np.any(point[:, 1] >= h)
        line = '|'.join(['image_%05d' % i,
                         ','.join(['%.3f' % min_x, '%.3f' % min_y, '%.3f' % max_x, '%.3f' % max_y]),
                         str(difficult)]) + '\n'
        lines.append(line)
        i += 1
        if i == 10000:
            break

    with open(os.path.join(annotations_root, 'ann.txt'), 'w') as f:
        f.writelines(lines)
