import numpy as np
import time
import matplotlib.pyplot as plt


def ed(m, n):
    m = m[1:-1]
    list1 = m.split(",")
    list1 = list(map(float, list1))

    n = n[1:-1]
    list2 = n.split(",")
    list2 = list(map(float, list2))

    return ((list1[0] - list2[0]) ** 2) + ((list1[1] - list2[1]) ** 2)


def pose_sub(datax, datay, num):
    #print('-------------------------------------')
    #print(datay)
    lenx = len(datax)
    sum_score = 0
    point_score = 100.0 / lenx
    for i in range(lenx):
        d_x = float(datax[i])

        d_y_1 = float(datay[i])
        dif = abs(d_x - d_y_1)

        # 设定一个值的上下界
        t = 0.2
        up_data = (1 + t) * d_x
        low_data = (1 - t) * d_x

        if low_data < d_y_1 < up_data:
            # 不变
            point_score = 100.0 / lenx
        elif dif > 1 * d_x:  # 超过0.7倍数得1分
            point_score = 0.0
        else:  # 0.1到0.7之间的扣占比分
            point_score = point_score * (1 - dif / d_x + t)
        sum_score = sum_score + point_score
    return sum_score


def pose_score(dis):
    sim = 1.0 / (1.0 + dis)
    score = sim * 100
    return score


def realize_dtw(x, y):
    lam = 1 / 4
    maxn = np.inf
    Num_col = x.shape[1]
    lenx = x.shape[0] + 1
    leny = y.shape[0] + 1
    # print(y)
    print('num_col', Num_col)
    print('lenx:', lenx)
    print('leny:', leny)

    print("比对中，请稍后......")
    dist = np.full((lenx, leny), 0, dtype=float);
    # 定义一个二维数组，初始化为0，数组类型为float
    for i in range(1, lenx):
        dist[i][0] = 0
    for i in range(1, leny):
        dist[0][i] = 0
    dist[0][0] = 0
    # print(dist[lenx-1][leny-1])
    for i in range(1, lenx):
        for j in range(1, leny):
            # print(x.iloc[i])
            tmp = pose_sub(x.iloc[i - 1], y.iloc[j - 1], Num_col)
            # print(pose_score(tmp))
            # dist[i][j] = tmp + min(dist[i - 1][j], dist[i - 1][j - 1], dist[i][j - 1])
            dist[i][j] = tmp
    pose_sum = 0
    # print(dist[lenx - 1][leny - 1])
    # print(pose_score(dist[lenx-1][leny-1]))

    for i in range(1, lenx):
        dist[i][0] = 0
    for i in range(1, leny):
        dist[0][i] = 0

    rx = int(lam * lenx)
    ry = int(lam * leny)
    k = (leny * 1.0) / (lenx * 1.0)
    end_rx = lenx - rx
    end_ry = leny - ry

    # 全局约束
    for i in range(end_rx):
        start = int(k * i + ry)
        for j in range(start, leny):
            dist[i][j] = 0

    for i in range(rx, lenx):
        end = int(k * i - ry)
        if end < 0:
            end = 0
        for j in range(end):
            dist[i][j] = 0
    '''
    for i in range(lenx):
        for j in range(leny):
            print(dist[i][j], end='|')
        print()
    '''

    path1 = []
    path2 = []

    a = lenx - 1
    b = leny - 1
    # print(dist[lenx-1][leny-1])
    cnt = 0

    redun = -1
    cnt_redun = 0
    while (a > 0 and b > 0):
        cnt = cnt + 1
        pose_sum = pose_sum + dist[a][b]
        redun = abs(b - a) / (lenx / 2)
        cnt_redun = cnt_redun + (redun)
        # print(a-1,end='|')
        # print(b-1)
        path1.append(a - 1)
        path2.append(b - 1)
        if a == 1:
            b = b - 1
            continue
        elif b == 1:
            a = a - 1
            continue
        pd = np.argmax((dist[a - 1][b - 1], dist[a - 1][b], dist[a][b - 1]))
        if pd == 0:
            a = a - 1
            b = b - 1
        elif pd == 1:
            a = a - 1
        elif pd == 2:
            b = b - 1
    dist1 = dist[1:, 1:]
    # print('cnt:', cnt)
    # print('cnt_redun:', cnt_redun)

    cnt_redun = int(cnt_redun)
    print('final score:', pose_sum / cnt)

    # return pose_sum / cnt
    # print('nice')

    plt.imshow(dist1.T, origin='lower', cmap='gray', interpolation='nearest')  # T是矩阵转置的意思
    plt.plot(path1, path2)
    plt.show()
