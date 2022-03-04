from bayes_optim.kpca import MyKernelPCA
from bayes_optim.mylogging import eprintf
import pytest


def test_kpca():
    X = [[6.888437030500963, 5.159088058806049], [-1.5885683833831, -4.821664994140733], [0.22549442737217085, -1.9013172509917133], [5.675971780695452, -3.933745478421451], [-0.4680609169528829, 1.6676407891006235]]
    X_weighted = [[1.079483540452395, 0.808478123348775], [-0.0, -0.0], [0.015436285850211205, -0.13015521900151383], [2.8024450976474777, -1.9422411099521695], [-0.1315704411634408, 0.4687685435317057]]
    Y = [35.441907931514024, 455.983619954143, 288.22967496622755, 26.86758082381718, 30.021247428418974]
    kpca = MyKernelPCA(X, 'rbf', epsilon=0.1, kernel_params_dict={'gamma': 0.01})
    kpca.fit(X_weighted)
    points = [[0.1, 4.3], [2.3, 3.2], [-1.2, 4.1], [0.97627008, 4.30378733]]
    ys = kpca.transform(points)
    eprintf("transformed point is", ys)
    points1 = kpca.inverse_transform(ys)
    for (point, point1) in zip(points, points1):
        assert point == pytest.approx(point1, 0.01) 


