import statistics
from kirill.utils import *
from sklearn.decomposition import PCA, KernelPCA


X, values, c = sample_doe(-5., 5., 2, 1000, bn.F17())

pca = PCA(n_components=2, svd_solver='full')
Y = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
variances = get_column_variances(Y)
print('variances', variances)
vp = get_sorted_var_columns_pairs(Y)
print('sorted variances pairs', vp)


kpca = KernelPCA(kernel='rbf', gamma=0.001)
Y = kpca.fit_transform(X)
vp = get_sorted_var_columns_pairs(Y)
print('sorted variances pairs', vp[0:10])


