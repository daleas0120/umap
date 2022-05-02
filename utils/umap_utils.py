import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics as metrics2
from skimage import metrics
import pandas as pd

def draw_umap(data, n_neighbors=15, min_dist=1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data)
    return u

def plot3D(m, n, labeledDataByClass, labeledDataByType):
    c1 = ['r', 'g', 'b', 'm', 'k', 'c', 'r', 'g', 'b', 'c', 'k']
    fig_1_name = '3d_Classes_N' + str(n) + '_dist' + str(m) + '.png'
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    color_counter = 0
    for class_name in labeledDataByClass['RW']:
        bp = 0
        if labeledDataByClass['RW'][class_name]['x'].__len__() > 0:
            ax.scatter(np.array(labeledDataByClass['RW'][class_name]['x']),
                       np.array(labeledDataByClass['RW'][class_name]['y']),
                       np.array(labeledDataByClass['RW'][class_name]['z']),
                       c=c1[color_counter], cmap="Dark2", s=16, label=class_name)
        if labeledDataByClass['VW'][class_name]['x'].__len__() > 0:
            ax.scatter(np.array(labeledDataByClass['VW'][class_name]['x']),
                       np.array(labeledDataByClass['VW'][class_name]['y']),
                       np.array(labeledDataByClass['VW'][class_name]['z']),
                       c=c1[color_counter], cmap="Dark2", s=16, label=class_name)
        color_counter += 1

    #plt.setp(ax, xticks=[], yticks=[])
    plt.title("UMAP 3D Embedding", fontsize=18)
    ax.legend(loc='lower left', fontsize=18)
    plt.tight_layout()
    plt.savefig(fig_1_name, dpi=100)
    plt.show()

    fig_2_name = '3d_RW_VW_N' + str(n) + '_dist' + str(m) + '.png'
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(projection='3d')

    ax1.scatter(np.array(labeledDataByType['VW']['x']),
                np.array(labeledDataByType['VW']['y']),
                np.array(labeledDataByType['VW']['z']),
                c=c1[2],
                cmap="Dark2", s=16, label='VW')

    ax1.scatter(np.array(labeledDataByType['RW']['x']),
                np.array(labeledDataByType['RW']['y']),
                np.array(labeledDataByType['RW']['z']),
                c=c1[0],
                cmap="Dark2", s=16, label='RW')

    plt.title("UMAP 3D Embedding", fontsize=18)
    ax1.legend(loc='lower left', fontsize=18)
    plt.tight_layout()
    plt.savefig(fig_2_name, dpi=100)
    plt.show()

def plot(m, n, labeledDataByClass, labeledDataByType):
    # Plot the data
    fig_1_name = 'UMAP2Dclass_' + str(n) + '_dist' + str(m) + '.png'
    c1 = ['r', 'g', 'b', 'm', 'k', 'c', 'r', 'g', 'b', 'c', 'k']
    fig, ax = plt.figure(figsize=(12, 10))

    color_counter = 0
    for class_name in labeledDataByClass['RW']:
        bp = 0
        if labeledDataByClass['RW'][class_name]['x'].__len__() > 0:
            ax.scatter(np.array(labeledDataByClass['RW'][class_name]['x']),
                       np.array(labeledDataByClass['RW'][class_name]['y']),
                       c=c1[color_counter], cmap="Dark2", s=16, label=class_name)
        if labeledDataByClass['VW'][class_name]['x'].__len__() > 0:
            ax.scatter(np.array(labeledDataByClass['VW'][class_name]['x']),
                       np.array(labeledDataByClass['VW'][class_name]['y']),
                       c=c1[color_counter], cmap="Dark2", s=16, label=class_name)
        color_counter += 1

    #plt.setp(ax, xticks=[], yticks=[])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("2D UMAP", fontsize=18)
    ax.legend(loc='lower left', fontsize=18)
    plt.tight_layout()
    plt.savefig(fig_1_name, dpi=100)
    plt.show()

    fig_2_name = 'RW_VW_N' + str(n) + '_dist' + str(m) + '.png'
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    ax1.scatter(np.array(labeledDataByType['VW']['x']), np.array(labeledDataByType['VW']['y']), c=c1[2],
                cmap="Dark2", s=16, label='VW')
    ax1.scatter(np.array(labeledDataByType['RW']['x']), np.array(labeledDataByType['RW']['y']), c=c1[0],
                cmap="Dark2", s=16, label='RW')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax1.legend(loc='lower left', fontsize=18)
    plt.tight_layout()
    plt.savefig(fig_2_name, dpi=100)
    plt.show()

def assignClusters(data, epsilon, neighbors):

    clustering = DBSCAN(eps=epsilon, min_samples=neighbors).fit(data)
    return clustering.labels

def clusterDBSCAN(embedding, labels_true_class):
    ## Cluster Points
    db = DBSCAN(eps=0.23, min_samples=6).fit(embedding)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    dbscanlabels = db.labels_
    df_dbscanlabels = pd.DataFrame(dbscanlabels, columns=['DBSCAN_ID'])


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(dbscanlabels)) - (1 if -1 in dbscanlabels else 0)
    n_noise_ = list(dbscanlabels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print("Homogeneity: %0.3f" % metrics2.homogeneity_score(labels_true_class, dbscanlabels))
    print("Completeness: %0.3f" % metrics2.completeness_score(labels_true_class, dbscanlabels))
    print("V-measure: %0.3f" % metrics2.v_measure_score(labels_true_class, dbscanlabels))
    print("Adjusted Rand Index: %0.3f" % metrics2.adjusted_rand_score(labels_true_class, dbscanlabels))
    print(
        "Adjusted Mutual Information: %0.3f"
        % metrics2.adjusted_mutual_info_score(labels_true_class, dbscanlabels)
    )
    #if n_clusters_ >=1 :
    #    print("Silhouette Coefficient: %0.3f" % metrics2.silhouette_score(embedding, dbscanlabels))
    #else:
    #    print('Cannot Find Silhouette Coeff: No clusters found')
    return dbscanlabels, df_dbscanlabels, n_clusters_, n_noise_

def plotDBSCAN(n, m, labels, embedding, n_clusters_, NUM_COMPONENTS):

    fig = plt.figure(figsize=(12, 10))
    dbscan_fig_name = 'dbscan_N' + str(n) + '_dist' + str(m) + '.png'
    if NUM_COMPONENTS >= 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.viridis(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):

        print('Plot Cluster ', k)
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        xy = embedding[labels == k]

        if k == -1:
            clusterName = 'Noise'
        else:
            clusterName = 'Cluster ' + str(k)

        if NUM_COMPONENTS == 3:
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], color=tuple(col), s=16, label=clusterName)
        else:
            ax.scatter(xy[:, 0], xy[:, 1], color=tuple(col), s=16, label=clusterName)
        plt.autoscale()

    if NUM_COMPONENTS >= 3:
        b = 0
        #ax.set_zlabel('Z')
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
    else:
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    plt.legend(fontsize=18)
    plt.title("DBSCAN Estimated number of clusters: %d" % n_clusters_, fontsize=18)
    plt.tight_layout()

    plt.savefig(dbscan_fig_name, dpi=100)
    plt.show()

def getHausdorffDist(n_clusters_, labels, embedding):
    for idx in range(n_clusters_):
        cluster1mask = labels == idx
        cluster1 = embedding[cluster1mask]

        for jdx in range(idx + 1, n_clusters_):
            cluster2mask = labels == jdx
            cluster2 = embedding[cluster2mask]

            # Call the Hausdorff function on the coordinates
            h_dist = metrics.hausdorff_distance(cluster1, cluster2)
            # hausdorff_point_a, hausdorff_point_b = \
            #     metrics.hausdorff_pair(cluster1, cluster2)

            print('Cluster ', idx, ' | Cluster ', jdx, ' | Hausdorff Distance: ', h_dist)
