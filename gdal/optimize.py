import gdal
import numpy as np
import cv2
from math import sqrt, pow, cos, sin, asin ,acos

def optimzie():
    import time
    t1 = time.time()
    image_path = '../demo_image/TEST_1m2_classify.tif'
    imd = cv2.imread(image_path, -1)
    layer1_index = np.where(imd==1)
    imd[imd==1]=255

    _,layer_1 = cv2.threshold(imd,127,255,cv2.THRESH_BINARY)

    ima_shape = (imd.shape[0],imd.shape[1],3)

    _,imd = layer

    new_im = np.zeros(ima_shape).astype(np.uint8)
    # layer1 =
    # new_im = new_im*255
    contours, hierarchy = cv2.findContours(layer_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    length = len(contours)

    for i in range(length):
        cnt = contours[i]
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        rc = cv2.minAreaRect(approx)
        po = cv2.boxPoints(rc)
        len_po = cv2.arcLength(po[:, np.newaxis, :], True)
        len_apporx = cv2.arcLength(approx, True)
        duan_b = min_len(po)
        print(duan_b, 'duanb')
        if duan_b == 0:
            continue
        delta = (len_apporx / len_po) * duan_b
        # po = po[:, np.newaxis, :]
        po = po.astype(np.int32)
        d2_po = np.squeeze(po)
        d2_approx = np.squeeze(approx)
        len_pol = d2_approx.shape[0]
        if len_pol < 4:
            # print(d2_approx)
            continue
        dengfen_rec = dengfen_pol(po, n=int(duan_b))
        # print(i)
        new_approx = np.empty((0, 2), dtype=np.int32)
        # print(len_pol,'mark')
        for k in range(len_pol):

            begin = k
            a = d2_approx[begin]

            if k == len_pol - 1:
                stop = 0
            else:
                stop = k + 1
            b = d2_approx[stop]
            try:
                out = dengfen_p(a, b, 15)
                distance, expect_ps = huas2(out, dengfen_rec, euclidean)
                new_a = a[np.newaxis, :]
                new_b = b[np.newaxis, :]
                # new_approx = np.insert(new_approx, 0, new_a, axis=0)
                # delta =0
                if euclidean(a, expect_ps[0]) != 0:
                    new_approx = np.append(new_approx, new_a, axis=0)
                    # print('test_suc')

                if distance < duan_b * 0.2:
                    # print(expect_ps)
                    # f = np.insert(d2_approx,begin+1,expect_ps,axis=0)
                    # d2_approx[begin,:]=expect_ps[0]
                    # d2_approx[stop,:]=expect_ps[-1]
                    # np.insert(d2_approx, top, expect_ps[-1])
                    # print(d2_approx)
                    # if
                    new_approx = np.append(new_approx, expect_ps, axis=0)
                if euclidean(a, expect_ps[-1]) != 0:
                    new_approx = np.append(new_approx, new_b, axis=0)

                # else:
                #     new_approx=np.append(new_approx,d2_approx[i],axis=0)
            except:
                continue
        try:
            new_d2 = d2_approx[:, np.newaxis, :]
            # new_im = mark_points(new_approx,new_im)
            news_approx = acute_angle(new_approx)
            new_d3 = new_approx[:, np.newaxis, :]
            new_d4 = cv2.approxPolyDP(new_d3, epsilon, True)
            cv2.polylines(new_im, [new_d4], True, (0, 0, 255), 2)
            print('work')
        except:
            continue
        # if i>50:
        #     break

    cv2.imwrite('haus7_tset.tif', new_im)
    delta_t = time.time() - t1
    print(delta_t)


def pologan(image_path, des_path):
    image_name = image_path.split('/')[-1]

    image_data = io.imread(image_path)
    chull = morphology.convex_hull_object(image_data, connectivity=1)
    io.imsave(des_path, chull)


def duobianx():
    image_path = r'C:\Users\EDZ\Desktop\building_regulation\bud__2048__2048.tif'
    des_path = './bud_out1.tif'
    imd = cv2.imread(image_path, -1)

    ret, binary = cv2.threshold(imd, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(imd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    print(length)
    img = np.zeros((2048, 2048, 3)).astype(np.uint8)

    # cv2.drawContours(img,)

    for i in range(length):
        cnt = contours[i]
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # approx = cv2.approxPolyDP(cnt)
        # cv2.drawContours(img, approx, -1, (0, 0, 255), 3)
        cv2.polylines(img, [approx], True, (255, 0, 255), 2)
    # cv2.imshow('1',img)
    # cv2.waitKey(0)
    cv2.imwrite('test2_d.tif', img)


def select_rectangel(img=None):
    image_path = r'C:\Users\EDZ\Desktop\building_regulation\bud__2048__2048.tif'
    imd = cv2.imread(image_path, -1)
    ima_shape = (2048, 2048, 3)

    new_im = np.zeros(ima_shape).astype(np.uint8)
    # new_im = new_im*255
    contours, hierarchy = cv2.findContours(imd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    length = len(contours)
    for i in range(length):
        cnt = contours[i]
        epsilon = 0.00001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        rc = cv2.minAreaRect(approx)
        po = cv2.boxPoints(rc)
        po = po[:, np.newaxis, :]
        po = po.astype(np.int32)

        cv2.polylines(new_im, [po], True, (0, 0, 255), 2)
        print(i)
        if i > 50:
            break
        # cv2.drawContours(img, po, -1, (0, 0, 255), 3)
        # cv2.imshow('{}'.format(i), new_im)
        # cv2.waitKey(10)
        # try:
        #     # new_im = cv2.polylines(new_im, po, True, (0, 0, 255), 2)
        #     # cv2.drawContours(img, approx, -1, (0, 0, 255), 3)
        #     # cv2.imshow('1',new_im)
        #     # cv2.waitKey(0)
        # except:
        #     continue
    # cv2.polylines(new_im, po, True, (0, 0, 255), 2)
    # cv2.imshow('1', new_im)
    cv2.imwrite('rect1.tif', new_im)
    # cv2.waitKey(111110)


def hausff():
    image_path = r'C:\Users\EDZ\Desktop\building_regulation\bud__2048__2048.tif'
    imd = cv2.imread(image_path, -1)
    ima_shape = (2048, 2048, 3)

    new_im = np.zeros(ima_shape).astype(np.uint8)
    # new_im = new_im*255
    contours, hierarchy = cv2.findContours(imd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    length = len(contours)

    for i in range(length):
        cnt = contours[i]
        epsilon = 0.00001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        rc = cv2.minAreaRect(approx)
        po = cv2.boxPoints(rc)
        # po = po[:, np.newaxis, :]
        po = po.astype(np.int32)
        d2_po = np.squeeze(po)
        d2_approx = np.squeeze(approx)
        ddd = po.ndim
        d = hausdorff_distance(d2_approx, d2_po, distance='euclidean')
        d2 = hausdorff_distance(d2_po, d2_approx, distance='euclidean')
        print(d, d2)
        cv2.polylines(new_im, [po], True, (0, 0, 255), 2)
    # np.random.seed(0)
    # X = np.random.random((1000,100))
    # Y = np.random.random((5000,100))
    #
    # # Test computation of Hausdorff distance with different base distances
    # print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='manhattan')}")
    # print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='euclidean')}")
    # print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='chebyshev')}")
    # print(f"Hausdorff distance test: {hausdorff_distance(X, Y, distance='cosine')}")


def min_len(rec):
    l1 = euclidean(rec[0, :], rec[1, :])
    l2 = euclidean(rec[1, :], rec[2, :])
    return min(l1, l2)


def re_rec(rec, po):
    dengfen_rec = dengfen_pol(rec)
    len_po = po.shape[0]
    len_pos = cv2.arcLength(po[:, np.newaxis, :], True)
    len_rec = cv2.arcLength(rec[:, np.newaxis, :], True)
    duan_b = min_len(rec)
    delta = (len_pos / len_rec) * duan_b
    # print(i)
    new_approx = np.empty((0, 2), dtype=np.int32)
    print(len_po, 'mark')
    for k in range(len_po):

        begin = k
        a = po[begin]

        if k == len_po - 1:
            stop = 0
        else:
            stop = k + 1
        b = po[stop]
        try:
            out = dengfen_p(a, b, 15)
            distance, expect_ps = huas2(out, dengfen_rec, euclidean)
            new_a = a[np.newaxis, :]
            new_b = b[np.newaxis, :]
            # new_approx = np.insert(new_approx, 0, new_a, axis=0)
            # delta =0

            new_approx = np.append(new_approx, new_a, axis=0)
            print('test_suc')

            if distance < delta:
                # print(expect_ps)
                # f = np.insert(d2_approx,begin+1,expect_ps,axis=0)
                # d2_approx[begin,:]=expect_ps[0]
                # d2_approx[stop,:]=expect_ps[-1]
                # np.insert(d2_approx, top, expect_ps[-1])
                # print(d2_approx)
                # if
                new_approx = np.append(new_approx, expect_ps, axis=0)
            if euclidean(a, expect_ps[-1]) != 0:
                new_approx = np.append(new_approx, new_b, axis=0)

            # else:
            #     new_approx=np.append(new_approx,d2_approx[i],axis=0)
        except:
            continue

    return new_app


def mark_points(a, im):
    len_p = a.shape[0]
    for i in range(len_p):
        color = (255, 0, 0)
        if i < 7:
            color = (0, 255, 0)
            im = cv2.putText(im, str(i), tuple(a[i]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=color,
                             thickness=1)
        else:
            im = cv2.putText(im, str(i), tuple(a[i]), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=color,
                             thickness=1)

    return im


def euclidean(array_x, array_y):
    n = array_x.shape[0]
    ret = 0.
    for i in range(n):
        ret += (array_x[i] - array_y[i]) ** 2
    return sqrt(ret)


def haus(XA, XB, distance_function):
    nA = XA.shape[0]
    nB = XB.shape[0]
    cmax = 0.
    # 记录替换点
    want_ps = []
    for i in range(nA):
        cmin = np.inf
        want_p = np.empty((0, 2), dtype=np.int32)
        for j in range(nB):
            # want_p = XB[j,:]
            d = distance_function(XA[i, :], XB[j, :])
            if d < cmin:
                cmin = d
                # 记录点的坐标，方便替换
                want_p = XB[j, :]

            if cmin < cmax:
                break
        want_ps.append(want_p)
        if cmin > cmax and np.inf > cmin:
            cmax = cmin
    return cmax, np.array(want_ps)


def cual_one2all(a, ps, distance_function):
    '''计算一个点到所有点的距离，并保留最大值'''
    n = ps.shape[0]
    all_p = []
    for i in range(n):
        d = distance_function(a, ps[i, :])
        all_p.append(d)
    all_p = np.array(all_p)
    index = np.argmin(all_p)
    return all_p[index], ps[index, :]


def huas2(XA, XB, distance_function):
    nA = XA.shape[0]
    nB = XB.shape[0]

    want_ps = []
    want_d = []
    for i in range(nA):
        d, p = cual_one2all(XA[i, :], XB, distance_function)
        want_d.append(d)
        want_ps.append(p)
    dd = max(want_d)

    return dd, np.array(want_ps)


def dengfen_pol(approx, n=55):
    '''approx  为2维'''
    len_p = approx.shape[0]
    all_slice = []
    for i in range(len_p):

        a = approx[i, :]
        if i == len_p - 1:
            b = approx[0, :]
        else:

            b = approx[i + 1, :]
        out = dengfen_p(a, b, n=n)
        all_slice.append(out)
    return np.concatenate(tuple(all_slice), axis=0)


# def dengfen_p(a,b,n):
#     ps = []
#     x = [0 for i in range(n)]
#     y = [0 for i in range(n)]
#     len_x = b[0] - a[0]
#     len_y = b[1]-a[1]
#     delta_x = len_x // n
#     delta_y = len_y //n
#     for i in range(n):
#         x[i] = i*delta_x + a[0]
#         if delta_x==0:
#             delta_y = (b[1]-a[1])//n
#             y[i]=a[1]+delta_y
#         else:
#             y[i]= a[1]-(a[0]-x[i])*(a[1]-b[1])/(a[0]-b[0])
#         p = [x[i],y[i]]
#         ps.append(p)

# return np.array(ps)

def dengfen_p(a, b, n):
    ps = []
    x = [0 for i in range(n)]
    y = [0 for i in range(n)]
    len_x = b[0] - a[0]
    len_y = b[1] - a[1]
    delta_x = len_x / n
    delta_y = len_y / n
    for i in range(n):
        x[i] = i * delta_x + a[0]
        y[i] = i * delta_y + a[1]
        p = [int(x[i]), int(y[i])]
        ps.append(p)
    return np.array(ps)


def acute_angle(array):
    '''array (n,2)'''

    p_nums = array.shape[0]
    assert p_nums > 2, 'number of points must greater than 2'
    rads = []
    for i in range(p_nums):
        if i == 0:
            left = p_nums - 1
        else:
            left = i - 1
        if i == p_nums - 1:
            right = 0
        else:
            right = i + 1
        a = euclidean(array[left, :], array[i, :])
        b = euclidean(array[i, :], array[right, :])
        c = euclidean(array[left, :], array[right, :])
        aco = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
        rad = acos(aco)
        rads.append(rad)
    return rads


if __name__=='__main__':
    optimzie()