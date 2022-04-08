from kirill.utils import *
from bayes_optim.kpca import MyKernelPCA


def separation():
    # Sample DoE
    dim = 2
    X, Y, _ = sample_doe(-5, 5, dim, 100, bn.F21())
    # Weighting scheme
    X_weighted = get_rescaled_points(X, Y)
    # Construct ПФ
    kpca = MyKernelPCA(0.1, X, {'kernel_name': 'rbf', 'kernel_parameters': {'gamma': 0.05}}, 10)
    # Find the value for C
    kpca.fit(X_weighted)
    C = 2 * (1 - kpca.kernel([0] * dim, [5] * dim))
    print(C)
    # Find O1 = ПФ(0)
    O1 = kpca.transform([0] * dim)[0]
    # Sample many points in the bounding box and out of it
    sampled_points, _, _ = sample_doe(-10, 10, dim, 10, bn.F21())
    # Check if the statement of out separation works
    cnt_checked = 0
    fails = 0
    outside = 0
    for p in sampled_points:
        p1 = kpca.transform(p)[0]
        d_hf_sqr = sum((p1i - origing1i)**2 for (p1i, origing1i) in zip(p1, O1))
        print(f'point = {p}, p1 = {p1}, O1 = {O1}, d_hf_sqr = {d_hf_sqr}')
        is_inside_bb = (sum([-5 <= pi <= 5 for pi in p])) == dim
        if not is_inside_bb:
            outside += 1
        if d_hf_sqr > C:
            cnt_checked += 1
            if is_inside_bb:
                fails += 1
                print(f'FAIL! point = {p}, p1 = {p1}, O1 = {O1}')
    print(f'checked = {cnt_checked}, failed = {fails}, outside = {outside}')


if __name__ == '__main__':
    random.seed(0)
    # ridge_experiment()
    # run_experiment1()
    # ridge_kernel_experiment()
    separation()

