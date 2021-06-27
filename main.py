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
FILE_PATH = 'C:/Users/daleas/Documents/umap/data/20210521T1453dataFrame.csv' #100% BG, small set
#FILE_PATH= 'C:/Users/daleas/Documents/umap/data/20210529T1502dataFrame.csv' #100% BG, KL=0; ; small set
#FILE_PATH = 'C:/Users/daleas/Documents/umap/data/20210603T1158dataFrame.csv' #0% BG; KL=0
#FILE_PATH = 'C:/Users/daleas/Documents/umap/data/20210604T1041dataFrame.csv' # 20% BG, KL = 0; lg set
#FILE_PATH = 'C:/Users/daleas/Documents/umap/data/20210608T1001dataFrame.csv' # 100% BG, KL=0; lg set

NUM_NEIGHBORS = 3
MIN_DISTANCE = 0.1

def draw_umap(data, n_neighbors=15, min_dist=1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    #
    # fig = plt.figure()
    # if n_components == 1:
    #     ax = fig.add_subplot(111)
    #     ax.scatter(u[:,0], range(len(u)), c=data)
    # if n_components == 2:
    #     ax = fig.add_subplot(111)
    #     ax.scatter(u[:,0], u[:,1], c=data)
    # if n_components == 3:
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(u[:,0], u[:,1], u[:,2], c=data, s=100)
    # plt.title(title, fontsize=18)
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

    classLabels = ["Plane", "Glider", "Kite", "Quadcopter", "Eagle"]
    dataTypesLabels = ["RW", "VW"]

    print('Running UMAP...')
    embedding = draw_umap(latentData_z, n_neighbors=NUM_NEIGHBORS, min_dist=MIN_DISTANCE, n_components=2, metric='euclidean', title='')
    np.savetxt('UMAPembedding.csv', embedding, delimiter=',')
    c1 = ['r', 'g', 'b', 'm', 'k']

    label1 = [classLabels[x] for x in latentData.Class.map({"Plane": 0, "Glider": 1, "Kite": 2, "Quadcopter":3, "Eagle":4})]
    label2 = [dataTypesLabels[x] for x in latentData.DataType.map({"RW": 0, "VW": 1})]

   # Sort the Data Into Classes

    labeledDataByClass = { "RW": {"Plane": {'x': [], 'y': []},
                                  "Glider": {'x': [], 'y': []},
                                  "Kite": {'x': [], 'y': []},
                                  "Quadcopter": {'x': [], 'y': []},
                                  "Eagle": {'x': [], 'y': []}},
                           "VW": {"Plane": {'x': [], 'y': []},
                                 "Glider":{'x': [], 'y': []},
                                 "Kite": {'x': [], 'y': []},
                                 "Quadcopter": {'x': [], 'y': []},
                                 "Eagle": {'x': [], 'y': []}}
                          }

    labeledDataByType = {"RW": {'x':[], 'y':[]}, "VW": {'x':[], 'y':[]}}

    for idx in range(0, label1.__len__()):
        labeledDataByClass[label2[idx]][label1[idx]]['x'].append(embedding[idx, 0])
        labeledDataByClass[label2[idx]][label1[idx]]['y'].append(embedding[idx, 1])


        labeledDataByType[label2[idx]]['x'].append(embedding[idx, 0])
        labeledDataByType[label2[idx]]['y'].append(embedding[idx, 1])

    # Plot the data
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']), c=c1[0],
               cmap="Dark2", s=16, label='Plane')
    ax.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']), c=c1[0],
               cmap="Dark2", s=16)

    ax.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']), c=c1[1],
               cmap="Dark2", s=16, label='Glider')
    ax.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']), c=c1[1],
               cmap="Dark2", s=16)

    ax.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']), c=c1[2],
               cmap="Dark2", s=16, label='Kite')
    ax.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']), c=c1[2],
               cmap="Dark2", s=16)

    ax.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']), np.array(labeledDataByClass['RW']['Quadcopter']['y']), c=c1[3],
               cmap="Dark2", s=16, label='Quadcopter')
    ax.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']), np.array(labeledDataByClass['VW']['Quadcopter']['y']), c=c1[3],
               cmap="Dark2", s=16)

    ax.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']), c=c1[4],
               cmap="Dark2", s=16, label='Eagle')
    ax.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']), c=c1[4],
               cmap="Dark2", s=16)

    plt.setp(ax, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax.legend(loc = 'lower left', fontsize=18)
    plt.savefig('fiveClasses.png', dpi=100)
    plt.show()

    fig1, ax1 = plt.subplots(figsize=(12, 10))
    ax1.scatter(np.array(labeledDataByType['VW']['x']), np.array(labeledDataByType['VW']['y']), c=c1[2],
               cmap="Dark2", s=16, label='VW')
    ax1.scatter(np.array(labeledDataByType['RW']['x']), np.array(labeledDataByType['RW']['y']), c=c1[0],
               cmap="Dark2", s=16, label='RW')
    plt.setp(ax1, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax1.legend(loc='lower left', fontsize=18)
    plt.savefig('RW_VW.png', dpi=100)
    plt.show()

    """ Set 2 of images"""

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']), c=c1[0],
               cmap="Dark2", s=16, label='RW Plane')
    ax2.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']), c=c1[2],
               cmap="Dark2", s=16, label='VW Plane')
    plt.setp(ax2, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax2.legend(loc = 'lower left', fontsize=18)
    plt.savefig('RW_VW_plane_v1.png', dpi=100)
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(12, 10))
    ax3.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']), c=c1[0],
               cmap="Dark2", s=16, label='RW Glider')
    ax3.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']), c=c1[2],
               cmap="Dark2", s=16, label='VW Glider')
    plt.setp(ax3, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax3.legend(loc='lower left', fontsize=18)
    plt.savefig('RW_VW_glider_v1.png', dpi=100)
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(12, 10))
    ax4.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']), c=c1[0],
               cmap="Dark2", s=16, label='RW Kite')
    ax4.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']), c=c1[2],
               cmap="Dark2", s=16, label='VW Kite')
    plt.setp(ax4, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax4.legend(loc='lower left', fontsize=18)
    plt.savefig('RW_VW_kite_v1.png', dpi=100)
    plt.show()

    fig5, ax5 = plt.subplots(figsize=(12, 10))
    ax5.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']), np.array(labeledDataByClass['RW']['Quadcopter']['y']), c=c1[0],
               cmap="Dark2", s=16, label='RW Quadcopter')
    ax5.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']), np.array(labeledDataByClass['VW']['Quadcopter']['y']), c=c1[2],
               cmap="Dark2", s=16, label='VW Quadcopter')
    plt.setp(ax5, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax5.legend(loc = 'lower left', fontsize=18)
    plt.savefig('RW_VW_quad_v1.png', dpi=100)
    plt.show()

    fig6, ax6 = plt.subplots(figsize=(12, 10))
    ax6.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']), c=c1[0],
               cmap="Dark2", s=16, label='RW Eagle')
    ax6.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']), c=c1[2],
               cmap="Dark2", s=16, label='VW Eagle')
    plt.setp(ax6, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax6.legend(loc = 'lower left', fontsize=18)
    plt.savefig('RW_VW_eagle_v1.png', dpi=100)
    plt.show()

    """ Set 3 of images"""
    fig7, ax7 = plt.subplots(figsize=(12, 10))

    ax7.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']),
               c=c1[4],
               cmap="Dark2", s=16, label='Other Data')
    ax7.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax7.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax7.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax7.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']),
               np.array(labeledDataByClass['RW']['Quadcopter']['y']),
                c=c1[4],
               cmap="Dark2", s=16)
    ax7.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']),
               np.array(labeledDataByClass['VW']['Quadcopter']['y']),
                c=c1[4],
               cmap="Dark2", s=16)
    ax7.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax7.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']),
               c=c1[4],
               cmap="Dark2", s=16)

    ax7.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']),
                c=c1[2],
                cmap="Dark2", s=16, label='VW Plane')
    ax7.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']),
                c=c1[0],
                cmap="Dark2", s=16, label='RW Plane')
    plt.setp(ax7, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax7.legend(loc='lower left', fontsize=18)
    plt.savefig('RW_VW_plane_v2.png', dpi=100)
    plt.show()

    fig8, ax8 = plt.subplots(figsize=(12, 10))
    ax8.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']),
               c=c1[4],
               cmap="Dark2", s=16, label='Other Classes')
    ax8.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']),
               c=c1[4],
               cmap="Dark2", s=16)

    ax8.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax8.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']),
               c=c1[4],
               cmap="Dark2", s=16)

    ax8.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']),
               np.array(labeledDataByClass['RW']['Quadcopter']['y']),
                c=c1[4],
               cmap="Dark2", s=16)
    ax8.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']),
               np.array(labeledDataByClass['VW']['Quadcopter']['y']),
                c=c1[4],
               cmap="Dark2", s=16)

    ax8.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax8.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']),
               c=c1[4],
               cmap="Dark2", s=16)

    ax8.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']),
               c=c1[2],
               cmap="Dark2", s=16, label='VW Glider')
    ax8.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']),
               c=c1[0],
               cmap="Dark2", s=16, label='RW Glider')

    plt.setp(ax8, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax8.legend(loc='lower left', fontsize=18)
    plt.savefig('RW_VW_glider_v2.png', dpi=100)
    plt.show()

    fig9, ax9 = plt.subplots(figsize=(12, 10))
    ax9.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']),
               c=c1[4],
               cmap="Dark2", s=16, label='Other Classes')
    ax9.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax9.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax9.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax9.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']),
               np.array(labeledDataByClass['RW']['Quadcopter']['y']),
                c=c1[4],
               cmap="Dark2", s=16)
    ax9.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']),
               np.array(labeledDataByClass['VW']['Quadcopter']['y']),
                c=c1[4],
               cmap="Dark2", s=16)
    ax9.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax9.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']),
               c=c1[4],
               cmap="Dark2", s=16)


    ax9.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']),
               c=c1[2],
               cmap="Dark2", s=16, label='VW Kite')
    ax9.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']),
               c=c1[0],
               cmap="Dark2", s=16, label='RW Kite')
    plt.setp(ax9, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax9.legend(loc='lower left', fontsize=18)
    plt.savefig('RW_VW_kite_v2.png', dpi=100)
    plt.show()

    fig10, ax10 = plt.subplots(figsize=(12, 10))
    ax10.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']),
               c=c1[4],
               cmap="Dark2", s=16, label='Other Classes')
    ax10.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']),
               c=c1[4],
               cmap="Dark2", s=16)

    ax10.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax10.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']),
               c=c1[4],
               cmap="Dark2", s=16)

    ax10.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax10.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax10.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax10.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']),
               c=c1[4],
               cmap="Dark2", s=16)

    ax10.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']),
               np.array(labeledDataByClass['VW']['Quadcopter']['y']),
                c=c1[2],
               cmap="Dark2", s=16, label='VW Quadcopter')
    ax10.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']),
               np.array(labeledDataByClass['RW']['Quadcopter']['y']),
                c=c1[0],
               cmap="Dark2", s=16, label='RW Quadcopter')
    plt.setp(ax10, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax10.legend(loc='lower left', fontsize=18)
    plt.savefig('RW_VW_quad_v2.png', dpi=100)
    plt.show()

    fig11, ax11 = plt.subplots(figsize=(12, 10))
    ax11.scatter(np.array(labeledDataByClass['RW']['Plane']['x']), np.array(labeledDataByClass['RW']['Plane']['y']),
               c=c1[4],
               cmap="Dark2", s=16, label='Other Classes')
    ax11.scatter(np.array(labeledDataByClass['VW']['Plane']['x']), np.array(labeledDataByClass['VW']['Plane']['y']),
               c=c1[4],
               cmap="Dark2", s=16)

    ax11.scatter(np.array(labeledDataByClass['RW']['Glider']['x']), np.array(labeledDataByClass['RW']['Glider']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax11.scatter(np.array(labeledDataByClass['VW']['Glider']['x']), np.array(labeledDataByClass['VW']['Glider']['y']),
               c=c1[4],
               cmap="Dark2", s=16)

    ax11.scatter(np.array(labeledDataByClass['RW']['Kite']['x']), np.array(labeledDataByClass['RW']['Kite']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax11.scatter(np.array(labeledDataByClass['VW']['Kite']['x']), np.array(labeledDataByClass['VW']['Kite']['y']),
               c=c1[4],
               cmap="Dark2", s=16)
    ax11.scatter(np.array(labeledDataByClass['RW']['Quadcopter']['x']),
               np.array(labeledDataByClass['RW']['Quadcopter']['y']),
                c=c1[4],
               cmap="Dark2", s=16)
    ax11.scatter(np.array(labeledDataByClass['VW']['Quadcopter']['x']),
               np.array(labeledDataByClass['VW']['Quadcopter']['y']),
                c=c1[4],
               cmap="Dark2", s=16)

    ax11.scatter(np.array(labeledDataByClass['VW']['Eagle']['x']), np.array(labeledDataByClass['VW']['Eagle']['y']),
               c=c1[2],
               cmap="Dark2", s=16, label='VW Eagle')
    ax11.scatter(np.array(labeledDataByClass['RW']['Eagle']['x']), np.array(labeledDataByClass['RW']['Eagle']['y']),
               c=c1[0],
               cmap="Dark2", s=16, label='RW Eagle')
    plt.setp(ax11, xticks=[], yticks=[])
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
    ax11.legend(loc='lower left', fontsize=18)
    plt.savefig('RW_VW_eagle_v2.png', dpi=100)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
