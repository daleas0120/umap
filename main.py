from sklearn.datasets import fetch_openml

import seaborn as sns
import pandas as pd
import yaml
import umap_utils as uu
import copy
import os
import json
from tqdm import tqdm, trange
import numpy as np

def main():
    # load config parameters
    #with open('config/IRdata_32D.yml') as file:
    #    config = yaml.load(file, Loader=yaml.FullLoader)

    #with open('config/circle.yml') as file:
    with open('config/RGBdata_32D.yml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    FILE_PATH = config['DATA']['FILE_PATH']
    LD_COLUMNS = config['UMAP_SETTINGS']['LD_COLUMNS']
    MIN_DISTANCE = config['UMAP_SETTINGS']['MIN_DISTANCE']
    NUM_NEIGHBORS = config['UMAP_SETTINGS']['NUM_NEIGHBORS']
    NUM_COMPONENTS = config['UMAP_SETTINGS']['NUM_COMPONENTS']
    CLASS_MAP = config['DATA']['CLASS_MAP']
    SUPERCLASS_MAP = config['DATA']['SUPERCLASS_MAP']
    CLASS_DICT = config['DATA']['CLASS_DICT']
    SUPERCLASS_DICT = config['DATA']['SUPERCLASS_DICT']
    CLASSES = config['DATA']['CLASSES']
    SUPERCLASSES = config['DATA']['SUPERCLASSES']

    # Set Plot Style
    sns.set(context="paper", style="white")

    # Import data as np.array, and convert to Pandas DataFrame
    print('Loading Data...')

    [saveDirPath, file] = os.path.split(FILE_PATH)

    saveDir = os.path.join(saveDirPath, 'umap')
    if not os.path.exists(saveDir):
        os.mkdir(saveDir)
    os.chdir(saveDir)

    latentData = pd.read_csv(FILE_PATH)

    latentData_z = latentData[LD_COLUMNS].values
    imgList = latentData.filename.values
    imgLabels1 = latentData.SuperClass.values
    df_imgLabels1 = pd.DataFrame(imgLabels1, columns=['SuperClass'])
    imgLabels2 = latentData.Class.values
    df_imgLabels2 = pd.DataFrame(imgLabels2, columns=['Class'])
    df_imgList = pd.DataFrame(imgList, columns=['Name'])
    labels_true_class = [SUPERCLASS_MAP[class_str] for class_str in imgLabels1]
    df_labels_true_class = pd.DataFrame(labels_true_class, columns=['GT_ID'])

    # Sort the Data Into Classes
    classLabels = copy.deepcopy(CLASSES)
    dataTypesLabels = copy.deepcopy(SUPERCLASSES)
    labeledDataByClass = copy.deepcopy(CLASS_DICT)
    labeledDataByType = copy.deepcopy(SUPERCLASS_DICT)


    label1 = [classLabels[x] for x in latentData.Class.map(CLASS_MAP)]
    label2 = [dataTypesLabels[x] for x in latentData.SuperClass.map(SUPERCLASS_MAP)]

    def see_OG_latent_space():
    ### Cluster without running UMAP
        for idx in range(0, label1.__len__()):
            labeledDataByClass[label2[idx]][label1[idx]]['x'].append(latentData_z[idx, 0])
            labeledDataByClass[label2[idx]][label1[idx]]['y'].append(latentData_z[idx, 1])

            labeledDataByType[label2[idx]]['x'].append(latentData_z[idx, 0])
            labeledDataByType[label2[idx]]['y'].append(latentData_z[idx, 1])

            if NUM_COMPONENTS == 3:
                labeledDataByClass[label2[idx]][label1[idx]]['z'].append(latentData_z[idx, 2])
                labeledDataByType[label2[idx]]['z'].append(latentData_z[idx, 2])

        if NUM_COMPONENTS == 3:
            uu.plot3D(0, 0, labeledDataByClass, labeledDataByType)
        else:
            uu.plot(0, 0, labeledDataByClass, labeledDataByType)

        labels, df_labels, n_clusters_, n_noise_, _ = \
            uu.clusterDBSCAN(latentData_z, imgLabels1)
    #see_OG_latent_space()

    ### ---- EMBEDDING BY CLASS ---- ###
    # Train on RW, embed the VW
    min = 0.0001
    num = 5
    train_data, test_data = uu.umap_by_class(latentData, num, min, NUM_COMPONENTS,
                                             'SuperClass', 'RW', 'SuperClass', 'VW', LD_COLUMNS)

    new_data = pd.concat([test_data, train_data])
    embedding = np.array(new_data.iloc[:, 7:])

    train_data_by_class, train_data_by_superClass = uu.format_data(train_data, CLASS_DICT, SUPERCLASS_DICT, CLASS_MAP,
                                                       SUPERCLASS_MAP, CLASSES, SUPERCLASSES, NUM_COMPONENTS)

    test_data_by_class, test_data_by_superClass = uu.format_data(test_data, CLASS_DICT, SUPERCLASS_DICT, CLASS_MAP,
                                                                 SUPERCLASS_MAP, CLASSES, SUPERCLASSES, NUM_COMPONENTS)

    new_data_by_class, new_data_by_superClass = uu.format_data(new_data, CLASS_DICT, SUPERCLASS_DICT, CLASS_MAP,
                                                               SUPERCLASS_MAP, CLASSES, SUPERCLASSES, NUM_COMPONENTS)

    if NUM_COMPONENTS >= 3:
        uu.plot3D(num, min, train_data_by_class, train_data_by_superClass)
        uu.plot3D(num, min, test_data_by_class, test_data_by_superClass)
        uu.plot3D(num, min, new_data_by_class, new_data_by_superClass)
    else:
        uu.plot(num, min, train_data_by_class, train_data_by_superClass)
        uu.plot(num, min, test_data_by_class, test_data_by_superClass)
        uu.plot3D(num, min, new_data_by_class, new_data_by_superClass)

    dbscanlabels, df_labels, n_clusters_, n_noise_, db_stats = \
        uu.clusterDBSCAN(embedding, labels_true_class, NUM_COMPONENTS)

    uu.plotDBSCAN(num, min, dbscanlabels, embedding, n_clusters_, NUM_COMPONENTS)

    cluster_distances = uu.getHausdorffDist(n_clusters_, dbscanlabels, embedding)

    metrics = {'h_dist': cluster_distances, 'metrics': db_stats}
    with open("crosstrain_umap_metrics.json", "w") as outfile:
        json.dump(metrics, outfile)

    df_embedding = pd.DataFrame(embedding)
    flatten = lambda nested_lists: [item for sublist in nested_lists for item in sublist]

    exportData = pd.concat([df_imgList.reset_index(drop=True),
                            df_imgLabels1.reset_index(drop=True),
                            df_imgLabels2.reset_index(drop=True),
                            df_embedding.reset_index(drop=True),
                            df_labels_true_class.reset_index(drop=True),
                            df_labels.reset_index(drop=True)],
                           axis=1, ignore_index=True)

    concat_colNames = [list(df_imgList.columns),
                       list(df_imgLabels1.columns),
                       list(df_imgLabels2.columns),
                       list(df_embedding.columns),
                       list(df_labels_true_class.columns),
                       list(df_labels.columns)
                       ]

    csv_file_name = 'UMAPembedding_N' + str(num) + '_dist' + str(min) + '.csv'
    json_file_name = "UMAPembedding_N"+str(num)+'_dist'+str(min)+'_metrics.json'

    export_filepath = os.path.join(saveDir, csv_file_name)
    exportData.to_csv(export_filepath)

    # Train on VW, embed the RW
    # train_data, test_data = uu.umap_by_class(latentData, 5, 0.01, NUM_COMPONENTS,
    #                                          'SuperClass', 'VW', 'SuperClass', 'RW', LD_COLUMNS)
    # new_data = pd.concat([train_data, test_data])
    #
    # train_data_by_class, train_data_by_superClass = uu.format_data(train_data, CLASS_DICT, SUPERCLASS_DICT, CLASS_MAP,
    #                                                    SUPERCLASS_MAP, CLASSES, SUPERCLASSES, NUM_COMPONENTS)
    #
    # test_data_by_class, test_data_by_superClass = uu.format_data(test_data, CLASS_DICT, SUPERCLASS_DICT, CLASS_MAP,
    #                                                              SUPERCLASS_MAP, CLASSES, SUPERCLASSES, NUM_COMPONENTS)
    #
    # new_data_by_class, new_data_by_superClass = uu.format_data(new_data, CLASS_DICT, SUPERCLASS_DICT, CLASS_MAP,
    #                                                            SUPERCLASS_MAP, CLASSES, SUPERCLASSES, NUM_COMPONENTS)
    #
    # if NUM_COMPONENTS >= 3:
    #     uu.plot3D(5, 0.01, train_data_by_class, train_data_by_superClass)
    #     uu.plot3D(5, 0.01, test_data_by_class, test_data_by_superClass)
    #     uu.plot3D(5, 0.01, new_data_by_class, new_data_by_superClass)
    # else:
    #     uu.plot(5, 0.01, train_data_by_class, train_data_by_superClass)
    #     uu.plot(5, 0.01, test_data_by_class, test_data_by_superClass)
    #     uu.plot3D(5, 0.01, new_data_by_class, new_data_by_superClass)


    ###---- Test different UMAP Combinations----###
    # for m in (pbar  := tqdm(MIN_DISTANCE)):
    #     pbar.set_description("Processing %s Distances" % m)
    #     for n in NUM_NEIGHBORS:
    #
    #         print('Running UMAP...')
    #         print("N=" + str(n) + ' M=' + str(m))
    #         embedding = uu.draw_umap(latentData_z,
    #                                  n_neighbors=n,
    #                                  min_dist=m,
    #                                  n_components=NUM_COMPONENTS,
    #                                  metric='euclidean', title='')
    #
    #         new_latent_data = copy.deepcopy(latentData)
    #         new_latent_data.iloc[:, 7:] = embedding
    #
    #         embedding_by_class, embedding_by_superclass = uu.format_data(new_latent_data, CLASS_DICT,
    #                                                                      SUPERCLASS_DICT, CLASS_MAP, SUPERCLASS_MAP,
    #                                                                      CLASSES, SUPERCLASSES, NUM_COMPONENTS)
    #
    #         if NUM_COMPONENTS >= 3:
    #             uu.plot3D(m, n, embedding_by_class, embedding_by_superclass)
    #         else:
    #             uu.plot(m, n, embedding_by_class, embedding_by_superclass)
    #
    #         df_embedding = pd.DataFrame(embedding)
    #         exportData = pd.concat([df_imgList.reset_index(drop=True),
    #                                 df_imgLabels1.reset_index(drop=True),
    #                                 df_imgLabels2.reset_index(drop=True),
    #                                 df_embedding.reset_index(drop=True),
    #                                 df_labels_true_class.reset_index(drop=True),
    #                                 df_labels.reset_index(drop=True)],
    #                                axis=1, ignore_index=True)
    #
    #         concat_colNames = [list(df_imgList.columns),
    #                            list(df_imgLabels1.columns),
    #                            list(df_imgLabels2.columns),
    #                            list(df_embedding.columns),
    #                            list(df_labels_true_class.columns),
    #                            list(df_labels.columns)
    #                            ]
    #
    #         flatten = lambda nested_lists: [item for sublist in nested_lists for item in sublist]
    #
    #         exportData.columns = flatten(concat_colNames)
    #
    #         csv_file_name = 'UMAPembedding_N' + str(n) + '_dist' + str(m) + '.csv'
    #         json_file_name = "UMAPembedding_N"+str(n)+'_dist'+str(m)+'_metrics.json'
    #
    #         export_filepath = os.path.join(saveDir, csv_file_name)
    #         exportData.to_csv(export_filepath)
    #
    #         dbscanlabels, df_dbscanlabels, n_clusters_, n_noise_, db_stats = \
    #             uu.clusterDBSCAN(embedding, labels_true_class)
    #
    #         ### Plotting results
    #         uu.plotDBSCAN(n, m, dbscanlabels, embedding, n_clusters_, NUM_COMPONENTS)
    #
    #         ### Calculate hausdorff distances for each cluster
    #         cluster_distances = uu.getHausdorffDist(n_clusters_, dbscanlabels, embedding)
    #
    #         metrics = {'h_dist': cluster_distances, 'metrics': db_stats}
    #         with open(json_file_name, "w") as outfile:
    #             json.dump(metrics, outfile)
    #
    #         # uu.plot(m, n)
    #

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
