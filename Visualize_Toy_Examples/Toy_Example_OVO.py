
import numpy as np
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from mlxtend.plotting import plot_decision_regions

def plot_data(X, y, ax, colors):

    num_classes = np.size(np.unique(y))

    # color current class
    for a in range(0, num_classes):
        t = np.argwhere(y == a)
        t = t[:, 0]
        ax.scatter(X[t, 0], X[t, 1], s=50, color=colors[a], edgecolor='k', linewidth=1.0)
    return




def plot_ovo_sep(estimator,X, y, minx, maxx, ax, colors):

    estimator.fit(X, y)
    number_of_classifiers = len(estimator.coef_)
    for j in range(number_of_classifiers):
        for i in range(j + 1):
            w = estimator.coef_[i]
            w = [w[0], w[1] + 0.0001]  # adding 0.0001 just to make sure we
            # don't devide by 0

            r = np.linspace(minx, maxx, 200)
            z = - w[0] / w[1] * r - estimator.intercept_[i] / w[1]

            ax.plot(r, z, linewidth=1.5, color=colors[i], zorder=6)
            ax.plot(r, z, linewidth=2.0, color='k', zorder=2)

def plot_decision_surface_(estimator, X, y, colors_surf, colors_line):
    minx = min(min(X[:, 0]), min(X[:, 1]))
    maxx = max(max(X[:, 0]), max(X[:, 1]))
    gapx = (maxx - minx) * 0.1
    minx -= gapx
    maxx += gapx

    fig = plt.figure(figsize=(10, 8))

    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 3, 1])

    # setup current axis

    ax = plt.subplot(gs[1], aspect='equal');
    plot_ovo_sep(estimator=estimator, X=X, y=y, minx=minx, maxx=maxx, ax=ax, colors=colors_line)
    y_s = np.array(y, dtype='int32')
    countourf_kwards = {'alpha': 0.2}
    scatter_kwargs = {'s': 60, 'edgecolor': 'k', 'alpha': None, 'marker': 'o'}
    scatter_highlight_kwargs = {'s': 120, 'label': None, 'marker': 'o', 'alpha': 0.7}
    plot_decision_regions(X=X, y=y_s, clf=estimator, contourf_kwargs=countourf_kwards, scatter_kwargs=scatter_kwargs,
                          scatter_highlight_kwargs=scatter_highlight_kwargs, legend=0, markers='o', colors=colors_surf)
    # plot_data(ax)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(minx, maxx)
    ax.axis('off')

    plt.show()

def grab_data(num_classes, case):
    if num_classes == 3:

        data = np.array([
            [0.14862, 0.081797, 0.20622, 0.23157, 0.33525, 0.2477,
             0.1371, 0.32373, 0.28226, 0.18318, 0.76613, 0.68088,
             0.91359, 0.79839, 0.71544, 0.58871, 0.71774, 0.93894,
             0.77765, 0.88134, 0.4182, 0.34217, 0.4735, 0.57949,
             0.4159, 0.52419, 0.32834, 0.25461, 0.197, 0.59332],
            [0.71574, 0.62536, 0.63703, 0.86152, 0.76822, 0.78571,
             0.85569, 0.65743, 0.55539, 0.55539, 0.89067, 0.76531,
             0.68367, 0.54665, 0.54665, 0.58746, 0.42128, 0.55831,
             0.70991, 0.76822, 0.18222, 0.27259, 0.28717, 0.053936,
             0.021866, 0.15889, 0.16472, 0.31341, 0.20262, 0.23178]])

        y = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.,
             1., 1., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])

        data[0][:10] = data[0][:10] + 0.5
        data[1][:10] = data[1][:10] + 0.5

        data[0][10:20] = data[0][10:20] + 0.4
        data[1][10:20] = data[1][10:20] - 0.2

        data[0][20:30] = data[0][20:30] - 0.1
        data[1][20:30] = data[1][20:30] + 0.15

    elif num_classes == 4:
        data2 = np.array([[0.12558, 0.077189, 0.17166, 0.29608, 0.3606, 0.21313,
                           0.45968, 0.33295, 0.46889, 0.21313, 0.39055, 0.52419,
                           0.43894, 0.58641, 0.60945, 0.70392, 0.73618, 0.7523,
                           0.68548, 0.61175, 0.2477, 0.49885, 0.37903, 0.27765,
                           0.16014, 0.14631, 0.30991, 0.42972, 0.48733, 0.53802,
                           0.43664, 0.66244, 0.77304, 0.78917, 0.65092, 0.55876,
                           0.5818, 0.64862, 0.74539, 0.8606],
                          [0.75948, 0.6312, 0.45627, 0.60204, 0.79446, 0.86443,
                           0.59329, 0.47085, 0.80612, 0.69534, 0.6691, 0.73615,
                           0.47668, 0.50583, 0.64869, 0.83528, 0.55539, 0.70991,
                           0.41837, 0.8965, 0.50875, 0.52041, 0.38921, 0.31633,
                           0.34257, 0.16472, 0.15015, 0.2551, 0.39796, 0.19971,
                           0.12391, 0.51458, 0.46793, 0.27551, 0.25802, 0.39213,
                           0.30466, 0.12099, 0.12974, 0.37464]])
        y=np.array([0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 1., 1.,1., 1., 1., 1., 1., 1.,
                    1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2.,3., 3., 3., 3., 3., 3.,
                    3., 3., 3., 3.])

        if case == 'ideal':
            data2[0][:10] = data2[0][:10] - 0.2
            data2[1][:10] = data2[1][:10] + 0.5

            # Red
            data2[0][10:20] = data2[0][10:20] + 0.5
            data2[1][10:20] = data2[1][10:20] + 0.4

            # Green
            data2[0][20:30] = data2[0][20:30] + 0.6
            data2[1][20:30] = data2[1][20:30] - 0.4

            # Purple

            data2[0][30:40] = data2[0][30:40] - 0.7
            data2[1][30:40] = data2[1][30:40] - 0.2
        elif case == 'normal':
            data2[0][:10] = data2[0][:10] - 0.2
            data2[1][:10] = data2[1][:10] + 0.5

            # Red
            data2[0][10:20] = data2[0][10:20] + 0.5
            data2[1][10:20] = data2[1][10:20] + 0.4

            # Green
            data2[0][20:30] = data2[0][20:30] + 0.3
            data2[1][20:30] = data2[1][20:30] + 0.1

            # Purple

            data2[0][30:40] = data2[0][30:40] - 0.7
            data2[1][30:40] = data2[1][30:40] - 0.3
        elif case == 'extreme':
            # Blue
            data2[0][:10] = data2[0][:10] - 0.2
            data2[1][:10] = data2[1][:10] + 0.5

            # Red
            data2[0][10:20] = data2[0][10:20] +0.5
            data2[1][10:20] = data2[1][10:20] +0.4

            # Green
            data2[0][20:30] = data2[0][20:30] -0.3
            data2[1][20:30] = data2[1][20:30] +0.2

            # Purple

            data2[0][30:40] = data2[0][30:40] -0.7
            data2[1][30:40] = data2[1][30:40] -0.3

        else:

            raise AttributeError('Iput Error: Got {} but expected ideal,normal,extreme istead.!'.format(case))
        data = data2
    else:

        raise AttributeError('Input Error: Num of classes {} is not supported by this application.!'.format(num_classes))

    if num_classes == 3:
        colors_line = [[0, 0.2, 0.51], [0, 0.2, 0.51], [0.6, 0.1, 0.4]]
    elif num_classes == 4:
        colors_line = [[0, 0.2, 0.51], [0, 0.2, 0.51], [0, 0.2, 0.51], [0.6, 0.1, 0.4], [0.6, 0.1, 0.4],
                       [0.1, 0.65, 0.45]]
    else:
        colors_line=None

    return data.T, y, colors_line




def test():
    X, y, colors_line = grab_data(num_classes=4, case='reme')

    clf = SVC(kernel='linear')

    colors_surf = str('#003382,' + '#991A66,' + '#1AA673,' + '#FFB380')

    plot_decision_surface_(estimator=clf, X=X, y=y, colors_surf=colors_surf, colors_line = colors_line)

if __name__ == "__main__":
    test()