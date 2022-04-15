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

def plot3D(m, n, labeledDataByClass, labeledDataByType):
    c1 = ['r', 'g', 'b', 'm', 'k']
    fig_1_name = '3d_fiveClasses_N' + str(n) + '_dist' + str(m) + '.png'
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(np.array(labeledDataByClass['RW']['F16']['x']),
               np.array(labeledDataByClass['RW']['F16']['y']),
               np.array(labeledDataByClass['RW']['F16']['z']),
               c=c1[0],
               cmap="Dark2", s=16, label='F16')
    ax.scatter(np.array(labeledDataByClass['VW']['F16']['x']),
               np.array(labeledDataByClass['VW']['F16']['y']),
               np.array(labeledDataByClass['VW']['F16']['z']),
               c=c1[0],
               cmap="Dark2", s=16)

    ax.scatter(np.array(labeledDataByClass['RW']['A10']['x']),
               np.array(labeledDataByClass['RW']['A10']['y']),
               np.array(labeledDataByClass['RW']['A10']['z']),
               c=c1[1],
               cmap="Dark2", s=16, label='A10')

    ax.scatter(np.array(labeledDataByClass['VW']['A10']['x']),
               np.array(labeledDataByClass['VW']['A10']['y']),
               np.array(labeledDataByClass['VW']['A10']['z']),
               c=c1[1],
               cmap="Dark2", s=16)

    ax.scatter(np.array(labeledDataByClass['RW']['B52']['x']),
               np.array(labeledDataByClass['RW']['B52']['y']),
               np.array(labeledDataByClass['RW']['B52']['z']),
               c=c1[2],
               cmap="Dark2", s=16, label='B52')

    ax.scatter(np.array(labeledDataByClass['VW']['B52']['x']),
               np.array(labeledDataByClass['VW']['B52']['y']),
               np.array(labeledDataByClass['VW']['B52']['z']),
               c=c1[2],
               cmap="Dark2", s=16)

    ax.scatter(np.array(labeledDataByClass['RW']['C5']['x']),
               np.array(labeledDataByClass['RW']['C5']['y']),
               np.array(labeledDataByClass['RW']['C5']['z']),
               c=c1[3],
               cmap="Dark2", s=16, label='C5')

    ax.scatter(np.array(labeledDataByClass['VW']['C5']['x']),
               np.array(labeledDataByClass['VW']['C5']['y']),
               np.array(labeledDataByClass['VW']['C5']['z']),
               c=c1[3],
               cmap="Dark2", s=16)

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
    c1 = ['r', 'g', 'b', 'm', 'k']
    # Plot the data
    fig_1_name = 'fiveClasses_N' + str(n) + '_dist' + str(m) + '.png'

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(np.array(labeledDataByClass['RW']['F16']['x']), np.array(labeledDataByClass['RW']['F16']['y']),
               c=c1[0],
               cmap="Dark2", s=16, label='F16')
    ax.scatter(np.array(labeledDataByClass['VW']['F16']['x']), np.array(labeledDataByClass['VW']['F16']['y']),
               c=c1[0],
               cmap="Dark2", s=16)

    ax.scatter(np.array(labeledDataByClass['RW']['A10']['x']), np.array(labeledDataByClass['RW']['A10']['y']),
               c=c1[1],
               cmap="Dark2", s=16, label='A10')
    ax.scatter(np.array(labeledDataByClass['VW']['A10']['x']), np.array(labeledDataByClass['VW']['A10']['y']),
               c=c1[1],
               cmap="Dark2", s=16)

    ax.scatter(np.array(labeledDataByClass['RW']['B52']['x']), np.array(labeledDataByClass['RW']['B52']['y']), c=c1[2],
               cmap="Dark2", s=16, label='B52')
    ax.scatter(np.array(labeledDataByClass['VW']['B52']['x']), np.array(labeledDataByClass['VW']['B52']['y']), c=c1[2],
               cmap="Dark2", s=16)

    ax.scatter(np.array(labeledDataByClass['RW']['C5']['x']), np.array(labeledDataByClass['RW']['C5']['y']),
               c=c1[3],
               cmap="Dark2", s=16, label='C5')
    ax.scatter(np.array(labeledDataByClass['VW']['C5']['x']), np.array(labeledDataByClass['VW']['C5']['y']),
               c=c1[3],
               cmap="Dark2", s=16)

    #plt.setp(ax, xticks=[], yticks=[])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title("Latent Dimension Z embedded into two dimensions by UMAP", fontsize=18)
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

def assignClusters(data, epsilon, neighbors):

    clustering = DBSCAN(eps=epsilon, min_samples=neighbors).fit(data)
    return clustering.labels

def clusterDBSCAN(embedding, labels_true_class):
    ## Cluster Points
    db = DBSCAN(eps=1, min_samples=25).fit(embedding)
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
