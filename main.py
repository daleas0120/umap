# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import pandas as pd
import numpy as np

#FILE_PATH = 'C:/Users/daleas/PycharmProjects/UMAP/20210517T1447z_LD32_dataFrame.csv'
#FILE_PATH = 'C:/Users/daleas/Documents/umap/data/20210521T1453dataFrame.csv' #100% BG, small set
#FILE_PATH= 'C:/Users/daleas/Documents/umap/data/20210529T1502dataFrame.csv' #100% BG, KL=0; ; small set
#FILE_PATH = 'C:/Users/daleas/Documents/umap/data/20210603T1158dataFrame.csv' #0% BG; KL=0
#FILE_PATH = 'C:/Users/daleas/Documents/umap/data/20210604T1041dataFrame.csv' # 20% BG, KL = 0; lg set
#FILE_PATH = 'C:/Users/daleas/PycharmProjects/UMAP/umap/data/20210608T1001dataFrame.csv' # 100% BG, KL=0; lg set
#FILE_PATH = 'C:/Users/daleas/PycharmProjects/UMAP/umap/data/20210627T1447dataFrame.csv' #custom BCE; alpha=0
#FILE_PATH = 'C:/Users/daleas/PycharmProjects/UMAP/umap/data/20210627T1447dataFrame_a0.csv' #customBCE; alpha=0
#FILE_PATH = 'C:/Users/daleas/PycharmProjects/UMAP/umap/data/20210628T1846dataFrame_apt2.csv' #customBCE; alpha=0.2
#FILE_PATH = 'C:/Users/daleas/PycharmProjects/UMAP/umap/data/20210701T1259dataFrame_apt4.csv' #customBCE; alpha=0.4
#FILE_PATH = 'C:/Users/daleas/PycharmProjects/UMAP/umap/data/20210701T1800dataFrame_apt6.csv' #customBCE; alpha=0.6
#FILE_PATH = 'C:/Users/daleas/PycharmProjects/UMAP/umap/data/20210703T1936dataFrame_apt8.csv' #customBCE; alpha=0.8
#FILE_PATH = 'C:/Users/daleas/PycharmProjects/UMAP/umap/data/20210709T0203dataFrame_a1.csv' #customBCE; alpha=1
#FILE_PATH = 'D:/Ashley_ML/VAE/VAE/VAE_logs/20210730T0039/results_20210730T0150/20210730T0150dataFrame.csv' #first IR
#FILE_PATH = 'C:/Users/daleas/Documents/VAE/VAE_logs/20220408T0045/results_20220408T1126/20220408T1126dataFrame.csv' #first AI CITY DATA
FILE_PATH = 'C:/Users/daleas/Documents/VAE/VAE_logs/20220408T0045/results_20220408T1536/20220408T1536dataFrame.csv'
NUM_NEIGHBORS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
#NUM_NEIGHBORS = [40]

#MIN_DISTANCE = np.linspace(0.1, 1, 10, endpoint=True)
MIN_DISTANCE = [0.1]
NUM_COMPONENTS = 3

def draw_umap(data, n_neighbors=15, min_dist=1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    return u

def main():
    # Set Plot Style
    sns.set(context="paper", style="white")

    # Import data as np.array, and convert to Pandas DataFrame
    print('Loading Data...')
    latentData = pd.read_csv(FILE_PATH)

    # latentData_z = pd.DataFrame(latentData, columns=['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7',
    # 'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15'
    # 'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22', 'z23'
    # 'z24', 'z25', 'z26', 'z27', 'z28', 'z29', 'z30', 'z31'])

    latentData_z = latentData[['z0', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7',
    'z8', 'z9', 'z10', 'z11', 'z12', 'z13', 'z14', 'z15',
    'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22', 'z23',
    'z24', 'z25', 'z26', 'z27', 'z28', 'z29', 'z30', 'z31']].values

    # Sort the Data Into Classes
    #classLabels = ["Plane", "Glider", "Kite", "Quadcopter", "Eagle"]
    #classLabels=['F16', 'A10', 'B52', 'C5']
    #classLabels = ["yellow", "orange", "green", "gray", "red", "blue", "white",
 #                  "golden", "brown", "black", "purple", "pink"]

    classLabels = ["sedan", "suv", "van", "hatchback", "mpv",  "pickup", "bus",
                   "truck", "estate", "sportscar", "RV"]


    dataTypesLabels = ["RW", "VW"]
    # labeledDataByClass = { "RW": {"Plane": {'x': [], 'y': [], 'z': []},
    #                               "Glider": {'x': [], 'y': [], 'z': []},
    #                               "Kite": {'x': [], 'y': [], 'z': []},
    #                               "Quadcopter": {'x': [], 'y': [], 'z': []},
    #                               "Eagle": {'x': [], 'y': [], 'z': []}},
    #                        "VW": {"Plane": {'x': [], 'y': [], 'z': []},
    #                              "Glider":{'x': [], 'y': [], 'z': []},
    #                              "Kite": {'x': [], 'y': [], 'z': []},
    #                              "Quadcopter": {'x': [], 'y': [], 'z': []},
    #                              "Eagle": {'x': [], 'y': [], 'z': []}}
    #                       }
    # labeledDataByClass = {"RW": {"F16": {'x': [], 'y': [], 'z': []},
    #                              "A10": {'x': [], 'y': [], 'z': []},
    #                              "B52": {'x': [], 'y': [], 'z': []},
    #                              "C5": {'x': [], 'y': [], 'z': []}},
    #                       "VW": {"F16": {'x': [], 'y': [], 'z': []},
    #                              "A10": {'x': [], 'y': [], 'z': []},
    #                              "B52": {'x': [], 'y': [], 'z': []},
    #                              "C5": {'x': [], 'y': [], 'z': []}}
    #                       }

    labeledDataByClass = {"RW":{"yellow": {'x': [], 'y': [], 'z': []},
                                 "orange": {'x': [], 'y': [], 'z': []},
                                 "green": {'x': [], 'y': [], 'z': []},
                                 "gray": {'x': [], 'y': [], 'z': []},
                                 "red": {'x': [], 'y': [], 'z': []},
                                 "blue": {'x': [], 'y': [], 'z': []},
                                 "white": {'x': [], 'y': [], 'z': []},
                                 "golden": {'x': [], 'y': [], 'z': []},
                                 "brown": {'x': [], 'y': [], 'z': []},
                                 "black": {'x': [], 'y': [], 'z': []},
                                 "purple": {'x': [], 'y': [], 'z': []},
                                 "pink": {'x': [], 'y': [], 'z': []}},
                          "VW":{"yellow": {'x': [], 'y': [], 'z': []},
                                 "orange": {'x': [], 'y': [], 'z': []},
                                 "green": {'x': [], 'y': [], 'z': []},
                                 "gray": {'x': [], 'y': [], 'z': []},
                                 "red": {'x': [], 'y': [], 'z': []},
                                 "blue": {'x': [], 'y': [], 'z': []},
                                 "white": {'x': [], 'y': [], 'z': []},
                                 "golden": {'x': [], 'y': [], 'z': []},
                                 "brown": {'x': [], 'y': [], 'z': []},
                                 "black": {'x': [], 'y': [], 'z': []},
                                 "purple": {'x': [], 'y': [], 'z': []},
                                 "pink": {'x': [], 'y': [], 'z': []}}
                          }

    labeledDataByClass ={"RW": {'sedan': {'x': [], 'y': [], 'z': []},
                               'suv': {'x': [], 'y': [], 'z': []},
                               'van': {'x': [], 'y': [], 'z': []},
                               'hatchback': {'x': [], 'y': [], 'z': []},
                               'mpv': {'x': [], 'y': [], 'z': []},
                               'pickup': {'x': [], 'y': [], 'z': []},
                               'bus': {'x': [], 'y': [], 'z': []},
                               'truck': {'x': [], 'y': [], 'z': []},
                               'estate': {'x': [], 'y': [], 'z': []},
                               'sportscar': {'x': [], 'y': [], 'z': []},
                               'RV': {'x': [], 'y': [], 'z': []}},
                         "VW": {'sedan': {'x': [], 'y': [], 'z': []},
                               'suv': {'x': [], 'y': [], 'z': []},
                               'van': {'x': [], 'y': [], 'z': []},
                               'hatchback': {'x': [], 'y': [], 'z': []},
                               'mpv': {'x': [], 'y': [], 'z': []},
                               'pickup': {'x': [], 'y': [], 'z': []},
                               'bus': {'x': [], 'y': [], 'z': []},
                               'truck': {'x': [], 'y': [], 'z': []},
                               'estate': {'x': [], 'y': [], 'z': []},
                               'sportscar': {'x': [], 'y': [], 'z': []},
                               'RV': {'x': [], 'y': [], 'z': []}}
                         }


    labeledDataByType = {"RW": {'x':[], 'y':[], 'z':[]}, "VW": {'x':[], 'y':[], 'z':[]}}

    ## Colormap info
    numClasses = 12
    #c1 = ['r', 'g', 'b', 'm', 'k', 'c', 'r', 'g', 'b', 'm', 'k', 'c']
    #c1 = ['#b5a81b', '#e67700', '#00e632', '#787878', '#c20404', '#0414c2',
    #      '#ffffff', '#ad9421', '#663e02', '#000000', '#6017b3', '#ff17f0']

    c1 = ['#ff17f0', '#e67700','#b5a81b',  '#00e632', '#c20404', '#0414c2',
          '#ffffff', '#ad9421', '#663e02', '#000000', '#6017b3']

    #c1 = np.linspace(0, 1, num=numClasses)
    colorMap = "viridis"
    # label1 = [classLabels[x] for x in latentData.Class.map({'F16': 0,
    #                                                         'A10': 1,
    #                                                         'B52': 2,
    #                                                         'C5': 3})]
    label1 = [classLabels[x] for x in latentData.Class.map({"sedan" : 0,
                                                            "suv": 1,
                                                            "van": 2,
                                                            "hatchback":3,
                                                            "mpv": 4,
                                                            "pickup": 5,
                                                            "bus": 6,
                                                            "truck": 7,
                                                            "estate": 8,
                                                            "sportscar": 9,
                                                            "RV": 10})]
    #label1 = [classLabels[x] for x in latentData.Class.map({"Plane": 0, "Glider": 1, "Kite": 2, "Quadcopter":3, "Eagle":4})]
    label2 = [dataTypesLabels[x] for x in latentData.DataType.map({"RW": 0, "VW": 1})]

    for m in MIN_DISTANCE:
        for n in NUM_NEIGHBORS:
            print('Running UMAP...')
            print("N="+str(n)+' M='+str(m))
            embedding = draw_umap(latentData_z,
                                  n_neighbors=n,
                                  min_dist=m,
                                  n_components=NUM_COMPONENTS,
                                  metric='euclidean', title='')

            csv_file_name = 'UMAPembedding_N'+str(n)+'_dist'+str(m)+'.csv'
            np.savetxt(csv_file_name, embedding, delimiter=',')

            for idx in range(0, label1.__len__()):
                labeledDataByClass[label2[idx]][label1[idx]]['x'].append(embedding[idx, 0])
                labeledDataByClass[label2[idx]][label1[idx]]['y'].append(embedding[idx, 1])


                labeledDataByType[label2[idx]]['x'].append(embedding[idx, 0])
                labeledDataByType[label2[idx]]['y'].append(embedding[idx, 1])

                if NUM_COMPONENTS == 3:
                    labeledDataByClass[label2[idx]][label1[idx]]['z'].append(embedding[idx, 2])
                    labeledDataByType[label2[idx]]['z'].append(embedding[idx, 2])
                # class1 = labeledDataByClass['RW']['F16']
                # class2 = labeledDataByClass['VW']['F16']
                # class3 = labeledDataByClass['RW']['A10']
                # class4 = labeledDataByClass['VW']['A10']
                # class5 = labeledDataByClass['RW']['B52']
                # class6 = labeledDataByClass['VW']['B52']
                # class7 = labeledDataByClass['RW']['C5']
                # class8 = labeledDataByClass['VW']['C5']

                # class1 = labeledDataByClass['VW']["yellow"]
                # class2 = labeledDataByClass['VW']["orange"]
                # class3 = labeledDataByClass['VW']["green"]
                # class4 = labeledDataByClass['VW']["gray"]
                # class5 = labeledDataByClass['VW']["red"]
                # class6 = labeledDataByClass['VW']["blue"]
                # class7 = labeledDataByClass['VW']["white"]
                # class8 = labeledDataByClass['VW']["golden"]
                # class9 = labeledDataByClass['VW']["brown"]
                # class10 = labeledDataByClass['VW']["black"]
                # class11 = labeledDataByClass['VW']["purple"]
                # class12 = labeledDataByClass['VW']["pink"]

                class1 = labeledDataByClass['VW']['sedan']
                class2 = labeledDataByClass['VW']['suv']
                class3 = labeledDataByClass['VW']['van']
                class4 = labeledDataByClass['VW']['hatchback']
                class5 = labeledDataByClass['VW']['mpv']
                class6 = labeledDataByClass['VW']['pickup']
                class7 = labeledDataByClass['VW']['bus']
                class8 = labeledDataByClass['VW']['truck']
                class9 = labeledDataByClass['VW']['estate']
                class10 = labeledDataByClass['VW']['sportscar']
                class11 = labeledDataByClass['VW']['RV']


                data = [class1, class2, class3, class5, class6,
                        class7, class8, class10, class11]

            def plot3D(m, n, labels, data):
                fig_1_name = '3d_'+str(numClasses)+ '_' + str(n) + '_dist' + str(m) + '.png'
                fig = plt.figure(figsize=(12, 10))
                ax = fig.add_subplot(projection='3d')

                #labels = ["yellow", "orange", "green", "gray", "red", "blue", "white",
                #          "golden", "brown", "black", "purple", "pink"]

                for i in range(data.__len__()):
                    ax.scatter(np.array(data[i]['x']),
                               np.array(data[i]['y']),
                               np.array(data[i]['z']),
                               c=c1[i], cmap=colorMap, s=16, label=labels[i], alpha=0.3
                               )

                plt.setp(ax, xticks=[], yticks=[])
                plt.title("Latent Dimension Z embedded into three dimensions by UMAP", fontsize=18)
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

                plt.setp(ax1, xticks=[], yticks=[])
                plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                ax1.legend(loc='lower left', fontsize=18)
                plt.tight_layout()
                plt.savefig(fig_2_name, dpi=100)
                plt.show()

            plot3D(m, n, classLabels, data)

            def plot(m, n):
                # Plot the data
                fig_1_name = 'fiveClasses_N'+str(n)+'_dist'+str(m)+'.png'
                
                class1 = labeledDataByClass['RW']['F16']

                fig, ax = plt.subplots(figsize=(12, 10))
                ax.scatter(np.array(['x']), np.array(labeledDataByClass['RW']['Plane']['y']), c=c1[0],
                           cmap="Dark2", s=16, label='F16')
                ax.scatter(np.array(labeledDataByClass['VW']['F16']['x']), np.array(labeledDataByClass['VW']['Plane']['y']), c=c1[0],
                           cmap="Dark2", s=16)

                ax.scatter(np.array(labeledDataByClass['RW']['A10']['x']), np.array(labeledDataByClass['RW']['Glider']['y']), c=c1[1],
                           cmap="Dark2", s=16, label='A10')
                ax.scatter(np.array(labeledDataByClass['VW']['A10']['x']), np.array(labeledDataByClass['VW']['Glider']['y']), c=c1[1],
                           cmap="Dark2", s=16)

                ax.scatter(np.array(labeledDataByClass['RW']['B52']['x']), np.array(labeledDataByClass['RW']['Kite']['y']), c=c1[2],
                           cmap="Dark2", s=16, label='B52')
                ax.scatter(np.array(labeledDataByClass['VW']['B52']['x']), np.array(labeledDataByClass['VW']['Kite']['y']), c=c1[2],
                           cmap="Dark2", s=16)

                ax.scatter(np.array(labeledDataByClass['RW']['C5']['x']), np.array(labeledDataByClass['RW']['Quadcopter']['y']), c=c1[3],
                           cmap="Dark2", s=16, label='C5')
                ax.scatter(np.array(labeledDataByClass['VW']['C5']['x']), np.array(labeledDataByClass['VW']['Quadcopter']['y']), c=c1[3],
                           cmap="Dark2", s=16)

                plt.setp(ax, xticks=[], yticks=[])
                plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                ax.legend(loc='lower left', fontsize=18)
                plt.tight_layout()
                plt.savefig(fig_1_name, dpi=100)
                plt.show()

                fig_2_name = 'RW_VW_N'+str(n)+'_dist'+str(m)+'.png'
                fig1, ax1 = plt.subplots(figsize=(12, 10))
                ax1.scatter(np.array(labeledDataByType['VW']['x']), np.array(labeledDataByType['VW']['y']), c=c1[2],
                           cmap="Dark2", s=16, label='VW')
                ax1.scatter(np.array(labeledDataByType['RW']['x']), np.array(labeledDataByType['RW']['y']), c=c1[0],
                           cmap="Dark2", s=16, label='RW')
                plt.setp(ax1, xticks=[], yticks=[])
                plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                ax1.legend(loc='lower left', fontsize=18)
                plt.tight_layout()
                plt.savefig(fig_2_name, dpi=100)
                plt.show()

                """ Set 2 of images"""
                # fig_2_name = 'RW_VW_plane_v1_N'+str(n)+'_dist'+str(m)+'.png'
                # fig2, ax2 = plt.subplots(figsize=(12, 10))
                # ax2.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']), c=c1[0],
                #            cmap="Dark2", s=16, label='RW Plane')
                # ax2.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']), c=c1[2],
                #            cmap="Dark2", s=16, label='VW Plane')
                # plt.setp(ax2, xticks=[], yticks=[])
                # plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                # ax2.legend(loc = 'lower left', fontsize=18)
                # plt.tight_layout()
                # plt.savefig(fig_2_name, dpi=100)
                # plt.show()
                #
                # fig_3_name = 'RW_VW_glider_v1_N'+str(n)+'_dist'+str(m)+'.png'
                # fig3, ax3 = plt.subplots(figsize=(12, 10))
                # ax3.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']), c=c1[0],
                #            cmap="Dark2", s=16, label='RW Glider')
                # ax3.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']), c=c1[2],
                #            cmap="Dark2", s=16, label='VW Glider')
                # plt.setp(ax3, xticks=[], yticks=[])
                # plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                # ax3.legend(loc='lower left', fontsize=18)
                # plt.tight_layout()
                # plt.savefig(fig_3_name, dpi=100)
                # plt.show()
                #
                # fig_4_name = 'RW_VW_kite_v1_N'+str(n)+'_dist'+str(m)+'.png'
                # fig4, ax4 = plt.subplots(figsize=(12, 10))
                # ax4.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']), c=c1[0],
                #            cmap="Dark2", s=16, label='RW Kite')
                # ax4.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']), c=c1[2],
                #            cmap="Dark2", s=16, label='VW Kite')
                # plt.setp(ax4, xticks=[], yticks=[])
                # plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                # ax4.legend(loc='lower left', fontsize=18)
                # plt.tight_layout()
                # plt.savefig(fig_4_name, dpi=100)
                # plt.show()
                #
                # fig_5_name = 'RW_VW_quad_v1_N'+str(n)+'_dist'+str(m)+'.png'
                # fig5, ax5 = plt.subplots(figsize=(12, 10))
                # ax5.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']), np.array(labeledDataByClass['RW']['Quadcopter']['y']), c=c1[0],
                #            cmap="Dark2", s=16, label='RW Quadcopter')
                # ax5.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']), np.array(labeledDataByClass['VW']['Quadcopter']['y']), c=c1[2],
                #            cmap="Dark2", s=16, label='VW Quadcopter')
                # plt.setp(ax5, xticks=[], yticks=[])
                # plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                # ax5.legend(loc = 'lower left', fontsize=18)
                # plt.tight_layout()
                # plt.savefig(fig_5_name, dpi=100)
                # plt.show()
                #
                # fig_6_name = 'RW_VW_eagle_v1_N'+str(n)+'_dist'+str(m)+'.png'
                # fig6, ax6 = plt.subplots(figsize=(12, 10))
                # ax6.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']), c=c1[0],
                #            cmap="Dark2", s=16, label='RW Eagle')
                # ax6.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']), c=c1[2],
                #            cmap="Dark2", s=16, label='VW Eagle')
                # plt.setp(ax6, xticks=[], yticks=[])
                # plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                # ax6.legend(loc = 'lower left', fontsize=18)
                # plt.tight_layout()
                # plt.savefig(fig_6_name, dpi=100)
                # plt.show()
                #
                # """ Set 3 of images"""
                # fig_7_name = 'RW_VW_plane_v2_N'+str(n)+'_dist'+str(m)+'.png'
                # fig7, ax7 = plt.subplots(figsize=(12, 10))
                #
                # ax7.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16, label='Other Data')
                # ax7.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax7.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax7.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax7.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']),
                #            np.array(labeledDataByClass['RW']['Quadcopter']['y']),
                #             c=c1[4],
                #            cmap="Dark2", s=16)
                # ax7.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']),
                #            np.array(labeledDataByClass['VW']['Quadcopter']['y']),
                #             c=c1[4],
                #            cmap="Dark2", s=16)
                # ax7.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax7.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax7.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']),
                #             c=c1[2],
                #             cmap="Dark2", s=16, label='VW Plane')
                # ax7.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']),
                #             c=c1[0],
                #             cmap="Dark2", s=16, label='RW Plane')
                # plt.setp(ax7, xticks=[], yticks=[])
                # plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                # ax7.legend(loc='lower left', fontsize=18)
                # plt.tight_layout()
                # plt.savefig(fig_7_name, dpi=100)
                # plt.show()
                #
                # fig_8_name = 'RW_VW_glider_v2_N'+str(n)+'_dist'+str(m)+'.png'
                # fig8, ax8 = plt.subplots(figsize=(12, 10))
                # ax8.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16, label='Other Classes')
                # ax8.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax8.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax8.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax8.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']),
                #            np.array(labeledDataByClass['RW']['Quadcopter']['y']),
                #             c=c1[4],
                #            cmap="Dark2", s=16)
                # ax8.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']),
                #            np.array(labeledDataByClass['VW']['Quadcopter']['y']),
                #             c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax8.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax8.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax8.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']),
                #            c=c1[2],
                #            cmap="Dark2", s=16, label='VW Glider')
                # ax8.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']),
                #            c=c1[0],
                #            cmap="Dark2", s=16, label='RW Glider')
                #
                # plt.setp(ax8, xticks=[], yticks=[])
                # plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                # ax8.legend(loc='lower left', fontsize=18)
                # plt.tight_layout()
                # plt.savefig(fig_8_name, dpi=100)
                # plt.show()
                #
                # fig_9_name = 'RW_VW_kite_v2_N'+str(n)+'_dist'+str(m)+'.png'
                # fig9, ax9 = plt.subplots(figsize=(12, 10))
                # ax9.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16, label='Other Classes')
                # ax9.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax9.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax9.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax9.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']),
                #            np.array(labeledDataByClass['RW']['Quadcopter']['y']),
                #             c=c1[4],
                #            cmap="Dark2", s=16)
                # ax9.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']),
                #            np.array(labeledDataByClass['VW']['Quadcopter']['y']),
                #             c=c1[4],
                #            cmap="Dark2", s=16)
                # ax9.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax9.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                #
                #
                # ax9.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']),
                #            c=c1[2],
                #            cmap="Dark2", s=16, label='VW Kite')
                # ax9.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']),
                #            c=c1[0],
                #            cmap="Dark2", s=16, label='RW Kite')
                # plt.setp(ax9, xticks=[], yticks=[])
                # plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                # ax9.legend(loc='lower left', fontsize=18)
                # plt.tight_layout()
                # plt.savefig(fig_9_name, dpi=100)
                # plt.show()
                #
                # fig_10_name = 'RW_VW_quad_v2_N'+str(n)+'_dist'+str(m)+'.png'
                # fig10, ax10 = plt.subplots(figsize=(12, 10))
                # ax10.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16, label='Other Classes')
                # ax10.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax10.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax10.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax10.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax10.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax10.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax10.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax10.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']),
                #            np.array(labeledDataByClass['VW']['Quadcopter']['y']),
                #             c=c1[2],
                #            cmap="Dark2", s=16, label='VW Quadcopter')
                # ax10.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']),
                #            np.array(labeledDataByClass['RW']['Quadcopter']['y']),
                #             c=c1[0],
                #            cmap="Dark2", s=16, label='RW Quadcopter')
                # plt.setp(ax10, xticks=[], yticks=[])
                # plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                # ax10.legend(loc='lower left', fontsize=18)
                # plt.tight_layout()
                # plt.savefig(fig_10_name, dpi=100)
                # plt.show()
                #
                # fig_11_name = 'RW_VW_eagle_v2_N'+str(n)+'_dist'+str(m)+'.png'
                # fig11, ax11 = plt.subplots(figsize=(12, 10))
                # ax11.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16, label='Other Classes')
                # ax11.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax11.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax11.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax11.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax11.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']),
                #            c=c1[4],
                #            cmap="Dark2", s=16)
                # ax11.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']),
                #            np.array(labeledDataByClass['RW']['Quadcopter']['y']),
                #             c=c1[4],
                #            cmap="Dark2", s=16)
                # ax11.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']),
                #            np.array(labeledDataByClass['VW']['Quadcopter']['y']),
                #             c=c1[4],
                #            cmap="Dark2", s=16)
                #
                # ax11.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']),
                #            c=c1[2],
                #            cmap="Dark2", s=16, label='VW Eagle')
                # ax11.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']),
                #            c=c1[0],
                #            cmap="Dark2", s=16, label='RW Eagle')
                # plt.setp(ax11, xticks=[], yticks=[])
                # plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
                # ax11.legend(loc='lower left', fontsize=18)
                # plt.tight_layout()
                # plt.savefig(fig_11_name, dpi=100)
                # plt.show()

            # plot(m, n)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
