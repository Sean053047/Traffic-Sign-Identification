import cv2
import numpy as np
import os


def get_nonzero_mean_std(img, step=None, lower_value_lim=0):
    """return : Mean_std = ((b_mean , b_std),
    (g_mean , g_std),
    (r_mean , r_std))"""
    if img.ndim == 3:
        b = img[img[:, :, 0] > lower_value_lim][:, 0]
        g = img[img[:, :, 1] > lower_value_lim][:, 1]
        r = img[img[:, :, 2] > lower_value_lim][:, 2]
        if step == None:
            Mean_std = ((b.mean(), b.std()), (g.mean(), g.std()), (r.mean(), r.std()))
        else:
            Mean_std = (
                (b[::step].mean(), b[::step].std()),
                (g[::step].mean(), g[::step].std()),
                (r[::step].mean(), r[::step].std()),
            )
    elif img.ndim == 2:
        if step == None:
            Mean_std = (
                img[img > lower_value_lim].mean(),
                img[img > lower_value_lim].std(),
            )
        else:
            Mean_std = (
                img[img > lower_value_lim][::step].mean(),
                img[img > lower_value_lim][::step].std(),
            )
    return Mean_std


def white_second_processing(img, mask):
    white_copy = np.zeros_like(img)
    white_copy = cv2.bitwise_and(img, img, mask=mask)
    b_channel = white_copy[:, :, 0].astype(np.float32)
    r_channel = white_copy[:, :, 2].astype(np.float32)
    rb_diff = r_channel - b_channel
    rb_diff[rb_diff < 0] = 0
    Mean_std = get_nonzero_mean_std(rb_diff)
    if Mean_std[0] < 10:
        lim = (Mean_std[0] + Mean_std[1]) if Mean_std[1] < 10 else 255
    elif Mean_std[0] < 20:
        lim = (Mean_std[0] + 0.6 * Mean_std[1]) if Mean_std[1] < 10 else Mean_std[0]
    elif Mean_std[0] < 30:
        lim = Mean_std[0]
    elif Mean_std[0] < 50:
        lim = (Mean_std[0] - 0.8 * Mean_std[1]) if Mean_std[1] < 10 else Mean_std[0]
    else:
        lim = (Mean_std[0] - 2 * Mean_std[1]) if Mean_std[1] < 10 else Mean_std[0]

    mask[rb_diff > lim] = 0
    return mask


def get_white(img, white_lower_lim_value=100):
    """只取b_channel and g_channel  mask,具有代表行"""
    """白色區域之b channel 及 g channel 的數值會高於背景，利用此特性盡量取出接近白色之區域"""

    # 這三行的算法，可以取出，除了很藍色跟狠紅色以外的區域的遮罩
    rm_shadow_mask = (img > white_lower_lim_value).astype("uint8")
    rm_shadow_mask = cv2.cvtColor(rm_shadow_mask, cv2.COLOR_BGR2GRAY)
    ret, rm_shadow_mask = cv2.threshold(rm_shadow_mask * 255, 1, 255, cv2.THRESH_BINARY)
    rm_shadow = cv2.bitwise_and(img, img, mask=rm_shadow_mask)

    Mean_std = get_nonzero_mean_std(rm_shadow, 20)
    cp = rm_shadow.copy()
    b_channel = cp[:, :, 0]
    g_channel = cp[:, :, 1]

    b_channel[b_channel < Mean_std[0][0] + Mean_std[0][1] * 1.3] = 0
    g_channel[g_channel < Mean_std[1][0] + Mean_std[1][1] * 1.3] = 0

    b_channel[b_channel > 0] = 255
    g_channel[g_channel > 0] = 255

    # * 僅使用bg 作為代表性的mask
    bg = cv2.bitwise_and(b_channel, g_channel)

    return white_second_processing(img, bg)


def get_sign(img):
    red_ratio_lim = 0.008
    blue_ratio_lim = 0.007
    color_diff_allow_range = 20

    # * 先查看 b channel 與g＆r的差值，及 r channel 與g&b 的差值，若大於一定範圍，則判斷該區為 sign 具代表性的範圍。
    cal = img.copy().astype(np.int32)
    bg_diff = cal[:, :, 0] - cal[:, :, 1]
    br_diff = cal[:, :, 0] - cal[:, :, 2]
    rb_diff = cal[:, :, 2] - cal[:, :, 0]
    rg_diff = cal[:, :, 2] - cal[:, :, 1]

    red_justify = (
        30 if img[:, :, 2].mean() < 115 else 60 if img[:, :, 2].mean() > 150 else 50
    )
    rb_justify = -30 if img[:, :, 2].mean() < 115 else -40

    condition1 = (bg_diff > color_diff_allow_range - 15).astype("uint8")
    condition2 = (br_diff > color_diff_allow_range - 15).astype("uint8")
    b_lim = cv2.bitwise_and(condition1, condition2)

    condition1 = (rg_diff > color_diff_allow_range + red_justify).astype("uint8")
    condition2 = (rb_diff > color_diff_allow_range + red_justify + rb_justify).astype(
        "uint8"
    )
    r_lim = cv2.bitwise_and(condition1, condition2)

    # * 先決定藍色居多還是紅色居多
    rb_lim = b_lim if np.sum(b_lim) > np.sum(r_lim) else r_lim
    rb_name = "blue" if np.sum(b_lim) > np.sum(r_lim) else "red"
    rb_ratio_lim = blue_ratio_lim if np.sum(b_lim) > np.sum(r_lim) else red_ratio_lim

    rb_ratio = np.sum((rb_lim > 0)) / (rb_lim.size)
    # print(f'goal ratio : {rb_ratio_lim} |  calculate : {rb_ratio}')

    # * 判斷解讀出來的顏色的佔比是否夠大，如果不夠大，有可能是白色停止線
    result = get_white(img, 90) if rb_ratio < rb_ratio_lim else rb_lim * 255
    result_color = "white" if rb_ratio < rb_ratio_lim else rb_name

    return result, result_color


def align_img(img, mask, name: str, size: int = 200):
    # * img pre-processing
    if name != "white":
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    else:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 11))  # * 橫軸: 5 直軸: 11
        result = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k, iterations=2)

    contour, hierarchy = cv2.findContours(
        result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # * get the contour with maximum area
    area = 0
    index = -1
    for i, cnt in enumerate(contour, 0):
        a = cv2.contourArea(cnt)
        if a > area:
            index = i
            area = a
    if index == -1:
        print("failed to get maximum contour area")
        return np.zeros((size, size, 3), dtype=np.uint8)

    # * get minAreaRect of the contour
    cnt = contour[index]
    rect_info = cv2.minAreaRect(
        cnt
    )  # ? Return value of minAreaRect : ( (center_x, center_y) , (height, width), rotation angle)
    box = np.int0(cv2.boxPoints(rect_info))  # ? Get four end points of rectangle

    # * Sort the order of four end points of rectangle.
    sorted_box = list(box.astype("int32"))
    sorted_box = sorted(sorted_box, key=lambda k: k[0])
    sb1 = sorted(sorted_box[:2], key=lambda k: k[1])
    sb2 = sorted(sorted_box[2:4], key=lambda k: k[1])
    sorted_box = sb1 + sb2
    box = np.array(sorted_box).astype(
        "float32"
    )  # ! Perspective transform need np.array to be 'float32'.
    box[box < 0] = 0

    # * Perspective Transform
    dst_point = np.array([[0, 0], [0, size], [size, 0], [size, size]]).astype("float32")
    window = (size, size)
    Perspective_matrix = cv2.getPerspectiveTransform(box, dst_point)
    result = cv2.warpPerspective(img, Perspective_matrix, window)

    return result


if __name__ == "__main__":
    data_folder = "./Training_data/RAW_DATA"
    for folder in os.listdir(data_folder):
        print("start to process: ", folder)
        for file in os.listdir(os.path.join(data_folder, folder)):
            p = os.path.join(data_folder, folder, file)
            img = cv2.imread(p)
            mask, color = get_sign(img)
            align = align_img(img, mask, color)

            # cv2.imshow('origin', img)
            # cv2.imshow('mask', mask)
            # cv2.imshow('align', align)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            cv2.imwrite(os.path.join("./Training_data/DST_DATA", folder, file), align)
