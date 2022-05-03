from sklearn.datasets import fetch_openml

import seaborn as sns
import pandas as pd
import yaml
import umap_utils as uu
import copy
import os

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

    uu.get_label_names(CLASS_DICT)

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
        bp = 0
        #uu.plot(0, 0, labeledDataByClass, labeledDataByType)

    labels, df_labels, n_clusters_, n_noise_ = \
        uu.clusterDBSCAN(latentData_z, imgLabels1)

    #uu.plotDBSCAN(0, 0, labels, latentData_z, n_clusters_, NUM_COMPONENTS)
    #uu.getHausdorffDist(n_clusters_, labels, latentData_z)

    ### Test different UMAP Combinations
    for m in MIN_DISTANCE:
        for n in NUM_NEIGHBORS:
            ### Clear variables
            umaplabeledDataByClass = copy.deepcopy(CLASS_DICT)
            umaplabeledDataByType = copy.deepcopy(SUPERCLASS_DICT)

            print('Running UMAP...')
            print("N=" + str(n) + ' M=' + str(m))
            embedding = uu.draw_umap(latentData_z,
                                     n_neighbors=n,
                                     min_dist=m,
                                     n_components=NUM_COMPONENTS,
                                     metric='euclidean', title='')

            df_embedding = pd.DataFrame(embedding)
            if NUM_COMPONENTS == 3:
                df_embedding.columns = ['x', 'y', 'z']
            if NUM_COMPONENTS == 2:
                df_embedding.columns = ['x', 'y']

            for idx in range(0, label1.__len__()):
                umaplabeledDataByClass[label2[idx]][label1[idx]]['x'].append(embedding[idx, 0])
                umaplabeledDataByClass[label2[idx]][label1[idx]]['y'].append(embedding[idx, 1])

                umaplabeledDataByType[label2[idx]]['x'].append(embedding[idx, 0])
                umaplabeledDataByType[label2[idx]]['y'].append(embedding[idx, 1])

                if NUM_COMPONENTS >= 3:
                    umaplabeledDataByClass[label2[idx]][label1[idx]]['z'].append(embedding[idx, 2])
                    umaplabeledDataByType[label2[idx]]['z'].append(embedding[idx, 2])

            if NUM_COMPONENTS >= 3:
                uu.plot3D(m, n, umaplabeledDataByClass, umaplabeledDataByType)
            else:
                uu.plot(m, n, umaplabeledDataByClass, umaplabeledDataByType)

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

            flatten = lambda nested_lists: [item for sublist in nested_lists for item in sublist]

            exportData.columns = flatten(concat_colNames)

            csv_file_name = 'UMAPembedding_N' + str(n) + '_dist' + str(m) + '.csv'
            export_filepath = os.path.join(saveDir, csv_file_name)
            exportData.to_csv(export_filepath)

            dbscanlabels, df_dbscanlabels, n_clusters_, n_noise_ = \
                uu.clusterDBSCAN(embedding, labels_true_class)

            ### Plotting results
            uu.plotDBSCAN(n, m, dbscanlabels, embedding, n_clusters_, NUM_COMPONENTS)

            ### Calculate hausdorff distances for each cluster
            uu.getHausdorffDist(n_clusters_, dbscanlabels, embedding)

            # uu.plot(m, n)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
