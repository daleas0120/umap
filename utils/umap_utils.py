import umap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics as metrics2
from skimage import metrics
import pandas as pd
from tqdm import tqdm, trange
import copy
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

def format_data(embedding, CLASS_DICT, SUPERCLASS_DICT, CLASS_MAP, SUPERCLASS_MAP, CLASSES, SUPERCLASSES, NUM_COMPONENTS):
    classLabels = copy.deepcopy(CLASSES)
    dataTypesLabels = copy.deepcopy(SUPERCLASSES)
    umaplabeledDataByClass = copy.deepcopy(CLASS_DICT)
    umaplabeledDataBySuperClass = copy.deepcopy(SUPERCLASS_DICT)
    label1 = [classLabels[x] for x in embedding.Class.map(CLASS_MAP)]
    label2 = [dataTypesLabels[x] for x in embedding.SuperClass.map(SUPERCLASS_MAP)]

    df_embedding = pd.DataFrame(embedding.iloc[:, 7:])

    if NUM_COMPONENTS == 3:
        df_embedding.columns = ['x', 'y', 'z']
    if NUM_COMPONENTS == 2:
        df_embedding.columns = ['x', 'y']

    for idx in range(0, label1.__len__()):
        umaplabeledDataByClass[label2[idx]][label1[idx]]['x'].append(df_embedding.iloc[idx, 0])
        umaplabeledDataByClass[label2[idx]][label1[idx]]['y'].append(df_embedding.iloc[idx, 1])

        umaplabeledDataBySuperClass[label2[idx]]['x'].append(df_embedding.iloc[idx, 0])
        umaplabeledDataBySuperClass[label2[idx]]['y'].append(df_embedding.iloc[idx, 1])

        if NUM_COMPONENTS >= 3:
            umaplabeledDataByClass[label2[idx]][label1[idx]]['z'].append(df_embedding.iloc[idx, 2])
            umaplabeledDataBySuperClass[label2[idx]]['z'].append(df_embedding.iloc[idx, 2])

    return umaplabeledDataByClass, umaplabeledDataBySuperClass


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
        legend_label = False
        if labeledDataByClass['RW'][class_name]['x'].__len__() > 0:
            legend_label = True
            ax.scatter(np.array(labeledDataByClass['RW'][class_name]['x']),
                       np.array(labeledDataByClass['RW'][class_name]['y']),
                       np.array(labeledDataByClass['RW'][class_name]['z']),
                       c=c1[color_counter], cmap="Dark2", s=16, label=class_name)
        if labeledDataByClass['VW'][class_name]['x'].__len__() > 0:
            if legend_label == True:
                new_label = '_nolegend'
            else:
                new_label = class_name
            ax.scatter(np.array(labeledDataByClass['VW'][class_name]['x']),
                       np.array(labeledDataByClass['VW'][class_name]['y']),
                       np.array(labeledDataByClass['VW'][class_name]['z']),
                       c=c1[color_counter], cmap="Dark2", s=16, label=new_label)
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
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot()

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

def get_epsilon(dataset, num_neighbors):
    neighbors = NearestNeighbors(n_neighbors=num_neighbors)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(np.linspace(0, len(distances), len(distances), endpoint=True), distances)
    plt.show()

    x = range(0, len(distances))
    kneedle = KneeLocator(x, distances, S=1.0, curve="concave", direction="increasing")
    kneedle.plot_knee()
    plt.show()
    epsilon = kneedle.knee_y

    if epsilon == None:
        epsilon = 0.23
    return epsilon

def clusterDBSCAN(embedding, labels_true_class, NUM_DIM):
    db_stats = {}

    #epsilon = get_epsilon(embedding, 2*NUM_DIM)
    #sample_num = 2*NUM_DIM
    ## Cluster Points
    db = DBSCAN(eps=0.5, min_samples=50).fit(embedding)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    dbscanlabels = db.labels_
    df_dbscanlabels = pd.DataFrame(dbscanlabels, columns=['DBSCAN_ID'])


    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(dbscanlabels)) - (1 if -1 in dbscanlabels else 0)
    n_noise_ = list(dbscanlabels).count(-1)
    homogeneity = metrics2.homogeneity_score(labels_true_class, dbscanlabels)
    completeness = metrics2.completeness_score(labels_true_class, dbscanlabels)
    v_measure = metrics2.v_measure_score(labels_true_class, dbscanlabels)
    adj_rand_ind = metrics2.adjusted_rand_score(labels_true_class, dbscanlabels)
    adj_mut_info = metrics2.adjusted_mutual_info_score(labels_true_class, dbscanlabels)

    db_stats['n_clusters'] = n_clusters_
    print("Estimated number of clusters: %d" % n_clusters_)

    db_stats['n_noise'] = n_noise_
    print("Estimated number of noise points: %d" % n_noise_)

    db_stats['homogeneity'] = homogeneity
    print("Homogeneity: %0.3f" % homogeneity)

    db_stats['completeness'] = completeness
    print("Completeness: %0.3f" % completeness)

    db_stats['V-measure'] = v_measure
    print("V-measure: %0.3f" % v_measure)

    db_stats['adj_rand_ind'] = adj_rand_ind
    print("Adjusted Rand Index: %0.3f" % adj_rand_ind)

    db_stats['adj_mut_info'] = adj_mut_info
    print("Adjusted Mutual Information: %0.3f" % adj_mut_info)

    #if n_clusters_ >=1 :
    #    print("Silhouette Coefficient: %0.3f" % metrics2.silhouette_score(embedding, dbscanlabels))
    #else:
    #    print('Cannot Find Silhouette Coeff: No clusters found')
    return dbscanlabels, df_dbscanlabels, n_clusters_, n_noise_, db_stats

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

        if NUM_COMPONENTS >= 3:
            ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], color=tuple(col), s=16, label=clusterName)
        else:
            ax.scatter(xy[:, 0], xy[:, 1], color=tuple(col), s=16, label=clusterName)
    print(ax.azim)
    if NUM_COMPONENTS < 3:
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

    plt.legend(fontsize=18)
    plt.title("DBSCAN Estimated number of clusters: %d" % n_clusters_, fontsize=18)
    plt.tight_layout()
    ax.set_zlim(3, 5.5)
    plt.autoscale()
    plt.savefig(dbscan_fig_name, dpi=100)
    plt.show()

def getHausdorffDist(n_clusters_, labels, embedding):

    hausdorff_dict = {}

    for idx in trange(n_clusters_):
        cluster_distances_dict = {}
        cluster1mask = labels == idx
        cluster1 = embedding[cluster1mask]

        for jdx in range(idx + 1, n_clusters_):
            cluster2mask = labels == jdx
            cluster2 = embedding[cluster2mask]

            # Call the Hausdorff function on the coordinates
            h_dist = metrics.hausdorff_distance(cluster1, cluster2)
            # hausdorff_point_a, hausdorff_point_b = \
            #     metrics.hausdorff_pair(cluster1, cluster2)

            cluster_distances_dict[jdx] = h_dist

            #print('Cluster ', idx, ' | Cluster ', jdx, ' | Hausdorff Distance: ', h_dist)

        hausdorff_dict[idx] = cluster_distances_dict

    return hausdorff_dict

def umap_by_class(latentData, n, m, NUM_COMPONENTS,
                  train_col_name, train_col_val, test_col_name, test_col_val, ld_col):
    """
    Trains UMAP on one class and embeds data from another class using the trained UMAP
    DOES NOT DO DIMENSIONALITY REDUCTION

    :param latentData: pandas dataframe containing the original VAE embedding
    :param n: integer, number of neighbors to consider
    :param m: double, minimum distances
    :param NUM_COMPONENTS: integer, number of dimensions
    :param train_col_name: string, the column in latentData to mask
    :param train_col_val: string, the values in train_col_name which will be used to train UMAP
    :param test_col_name: string, the column in latentData to mask
    :param test_col_val: string, the values in test_col_name which will be embedded using the trained UMAP
    :param ld_col: list of column names that contain the VAE latent embedding in latentData
    :return: two pandas dataframes of the same format as latentData,
    one with train embedding and one with test embedding
    """

    train_class = latentData.loc[latentData[train_col_name] == train_col_val]
    train_data = train_class[ld_col].values
    trans = umap.UMAP(n_neighbors=n,
                      min_dist=m,
                      n_components=NUM_COMPONENTS,
                      metric='euclidean').fit(train_data)

    train_embedding = trans.embedding_
    train_class.iloc[:, 7:] = train_embedding

    test_class = latentData.loc[latentData[test_col_name] == test_col_val]
    test_data = test_class[ld_col].values
    test_embedding = trans.transform(test_data)
    test_class.iloc[:, 7:] = test_embedding

    return train_class, test_class

