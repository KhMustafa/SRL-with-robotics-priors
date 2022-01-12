"""
This class is responsible for plotting the results of the state representation learning algorithm. The functions of this
class are called from StateRepresentation_Continuous_V2.py file. The plotting results are written to a pdf file whose
name and directory are defined in the main function of the StateRepresentation_Continuous_V2.py script.
"""
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.colors
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import os

plottitle = True
titlefont = 10
import math


def closeAll():
    plt.close('all')


def saveMultipage(filename, figs=None, dpi=100):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


def plot_camera_observations(patchsize, observations, m=8, n=10, name='Agent observation Samples trough rgb camera'):
    # plt.ion()
    plt.figure(name)
    for i in range(m * n):
        plt.subplot(m, n, i + 1)
        plt.imshow(observations[i].reshape(patchsize[1], patchsize[0], 3), interpolation='nearest')
        plt.xticks([])
        plt.yticks([])
    plt.pause(0.1)


def plot_single_camera_observation(patchsize, observation, name='Current observation sample'):
    plt.figure(name)
    plt.imshow(observation.reshape(patchsize[1], patchsize[0], 3), interpolation='nearest')
    plt.pause(0.1)


def plot_single_laser_observation(laser_original, laser_decoded, xlim=[-500, 500], ylim=[-500, 500],
                                  name='Current laser sample'):
    # plt.ion()
    fig = plt.figure(name)
    ax1 = fig.add_subplot(111)

    lenght = len(laser_original)
    laser_original_x = []
    laser_original_y = []
    laser_decoded_x = []
    laser_decoded_y = []
    for i in range(lenght):
        orientation = 2 * 3.1415 * float(i) / float(lenght)
        laser_original_x.append(laser_original[i] * math.cos(orientation))
        laser_original_y.append(laser_original[i] * math.sin(orientation))
        laser_decoded_x.append(laser_decoded[i] * math.cos(orientation))
        laser_decoded_y.append(laser_decoded[i] * math.sin(orientation))

    original = ax1.scatter(laser_original_x, laser_original_y, s=7, marker='x', c="blue", linewidths=0.01)
    decoded = ax1.scatter(laser_decoded_x, laser_decoded_y, s=7, marker='x', c="red", linewidths=0.01)
    agent = ax1.scatter(0, 0, s=100, c="green", linewidths=1, marker="4")
    plt.legend((original, decoded), ('Original laser values', 'Decoded leaser values'))

    if plottitle:
        plt.title(name, {'fontsize': titlefont})

    plt.ylabel('Dimension 2')
    plt.xlabel('Dimension 1')

    axes = plt.gca()
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)

    plt.pause(0.1)


def getPCA(input, dim):
    pca = PCA(n_components=dim)
    return pca.fit_transform(input)


def plot_2d_colormap(xy, val, name, label, color='magma', plotgoal=False, clip_val=False):
    def clip_rewards(rewards):
        rewards[rewards > 0] = 0.3
        rewards[rewards < -2] = -2.1
        plt.pause(0.1)

        return rewards

    if clip_val:
        val = clip_rewards(val)

    # plt.ion()
    fig = plt.figure(name)
    ax1 = fig.add_subplot(111)
    im = ax1.scatter(xy[:, 0], xy[:, 1], s=9, c=val, cmap=color, linewidths=0.0)
    if plotgoal:
        ax1.scatter(xy[:, 3], xy[:, 4], s=10, c="red", linewidths=0.01)
    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.ylabel('Dimension 1')
    plt.xlabel('Dimension 2')
    fig.colorbar(im, label=label, ax=ax1)
    plt.pause(0.1)


def plot_3d_colormap(xyz, val, name, color='magma', xlim=0, ylim=0, clip_val=True):
    def clip_rewards(rewards):
        rewards[rewards > 0] = 0.3
        rewards[rewards < -2] = -2.1
        plt.pause(0.1)
        return rewards

    if clip_val:
        val = clip_rewards(val)

    fig = plt.figure(name)
    ax1 = fig.add_subplot(111, projection='3d')

    # Create Color Map
    colormap = plt.get_cmap(color)

    norm = matplotlib.colors.Normalize(vmin=min(val), vmax=max(val))
    im = ax1.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1, c=colormap(norm(val)), linewidths=0.001)

    if plottitle:
        plt.title(name, {'fontsize': titlefont})

    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.set_zlabel('Dimension 3')
    ax1.view_init(azim=90)

    axes = plt.gca()
    if xlim and ylim:
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)

    plt.pause(0.1)


def plot_area_colormap(xy, excluded, name, color='magma'):
    xy = np.mgrid[xy[0][0]:xy[1][0]:0.01, xy[0][1]:xy[1][1]:0.01].reshape(2, -1).T
    val = []
    for p in xy:
        if excluded[0][0] < p[0] < excluded[1][0] and excluded[0][1] < p[1] < excluded[1][1]:
            val.append(0.1)
        else:
            val.append(0.01)

    fig = plt.figure(name)
    ax1 = fig.add_subplot(111)
    im = ax1.scatter(xy[:, 0], xy[:, 1], s=9, c=val, cmap=color, linewidths=0.0)
    plt.ylabel('Dimension 1')
    plt.xlabel('Dimension 2')
    plt.pause(0.1)


def compare_points(xy1, xy2, name, legend, color='magma'):
    fig = plt.figure(name)
    ax1 = fig.add_subplot(111)
    val = range(len(xy1))

    ax1.scatter(xy1[:, 0], xy1[:, 1], s=10, c=val, cmap=color, linewidths=1)
    ax1.scatter(xy2[:, 0], xy2[:, 1], s=10, c=val, cmap=color, linewidths=1, marker="x")

    for i in range(len(xy1)):
        ax1.annotate(i, (xy1[i, 0], xy1[i, 1]))
        ax1.annotate(i, (xy2[i, 0], xy2[i, 1]))

    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.legend(legend)
    plt.ylabel('Dimension 1')
    plt.xlabel('Dimension 2')
    plt.pause(0.1)


def plot_trajectories(trajectories, name, legend, plotgoal):
    fig = plt.figure(name)
    ax1 = fig.add_subplot(111)

    for trajectory in trajectories:
        ax1.plot(trajectory[:, 0], trajectory[:, 1], linewidth=1)

    if plotgoal:
        ax1.scatter(trajectories[0][0, 3], trajectories[0][0, 4], s=100, color="red", linewidths=1)

    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.legend(legend)
    plt.ylabel('Dimension 1')
    plt.xlabel('Dimension 2')
    plt.pause(0.1)


def plot_priors_history(epoch_loss_vector, name='Loss history SRL network', folder='.././training_results'):
    timesteps = np.array(range(len(epoch_loss_vector)))

    # Visualize loss history
    plt.figure(name)
    plt.plot(timesteps, epoch_loss_vector[:, 0], linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 1], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 2], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 3], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 4], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 5], '--', linewidth=1)
    plt.grid()
    plt.legend(['Training Loss', 'Temp coherence loss', 'Causality loss', 'Proportionality loss', 'Repeatability loss',
                'Regularization loss'])
    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.pause(0.1)

    saveMultipage(os.path.join(folder, "Priors_losses.pdf"))


def plot_priorsAE_history(epoch_loss_vector, name='Loss history SRL network', folder='.././training_results'):
    timesteps = np.array(range(len(epoch_loss_vector)))

    # Visualize loss history
    plt.figure(name)
    plt.plot(timesteps, epoch_loss_vector[:, 0], linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 1], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 2], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 3], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 4], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 5], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 6], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 7], '--', linewidth=1)
    plt.grid()
    plt.legend(['Training Loss', 'Temp coherence loss', 'Causality loss', 'Proportionality loss', 'Repeatability loss',
                'Camera Reconstruction loss', 'Laser Reconstruction loss', 'Regularization loss'])
    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.ylim((0, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.pause(0.1)

    saveMultipage(os.path.join(folder, "PriorsAE_losses.pdf"))


def plot_5priors_history(epoch_loss_vector, name='Loss history SRL network', folder='.././training_results'):
    timesteps = np.array(range(len(epoch_loss_vector)))

    # Visualize loss history
    plt.figure(name)
    plt.plot(timesteps, epoch_loss_vector[:, 0], linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 1], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 2], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 3], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 4], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 5], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 7], '--', linewidth=1)
    plt.grid()
    plt.legend(['Training Loss', 'Temp coherence loss', 'Causality loss', 'Proportionality loss', 'Repeatability loss',
                'Landmark loss', 'Regularization loss'])
    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.ylim((0, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.pause(0.1)

    saveMultipage(os.path.join(folder, "Priors_landmark_losses.pdf"))


def plot_AE_history(epoch_loss_vector, name='Loss history SRL network', folder='.././training_results'):
    timesteps = np.array(range(len(epoch_loss_vector)))

    # Visualize loss history
    # plt.ion()
    plt.figure(name)
    plt.plot(timesteps, epoch_loss_vector[:, 0], linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 1], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 2], '--', linewidth=1)
    plt.plot(timesteps, epoch_loss_vector[:, 3], '--', linewidth=1)
    plt.grid()
    plt.legend(['Training Loss', 'Camera Reconstruction loss', 'Laser Reconstruction loss', 'Regularization loss'])
    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.pause(0.1)

    saveMultipage(os.path.join(folder, "AE_losses.pdf"))


def plot_history(hist, name, yaxis, xaxis="Episode"):
    timesteps = np.array(range(len(hist)))

    # Visualize loss history
    # plt.ion()
    plt.figure(name)
    plt.plot(timesteps, hist, linewidth=1)
    plt.grid()
    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.pause(0.1)


def plot_multiple_history(hist, name, legend, yaxis, xaxis="Episode", startat=0):
    # Visualize loss history
    plt.figure(name)
    for hists in hist:
        hists = hists[startat:]
        timesteps = np.array(range(startat, startat + len(hists)))
        plt.plot(timesteps, hists, linewidth=1)

    plt.legend(legend)
    plt.grid()
    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.pause(0.1)


def plot_bar(values, name, xaxis, yaxis):
    timesteps = range(len(values))

    plt.figure(name)
    plt.bar(timesteps, values)
    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.pause(0.1)


def plot_mean_dev(mean, dev, name, xaxis, yaxis):
    timesteps = range(len(mean))

    plt.figure(name)
    plt.plot(timesteps, mean, 'k', color='#FF0000')
    plt.fill_between(timesteps, mean - dev, mean + dev, alpha=1, edgecolor='#fda3a3', facecolor='#fda3a3', linewidth=0)
    if plottitle:
        plt.title(name, {'fontsize': titlefont})
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.pause(0.1)


def moving_average(data_set, periods=5, multipleaxis=False):
    weights = np.ones(periods) / periods
    if multipleaxis:
        result = []
        for i in range(len(data_set)):
            result.append(np.convolve(data_set[i], weights, mode='valid'))
    else:
        result = np.convolve(data_set, weights, mode='valid')
    return result


def plot_all_srl(states, rews, data_dick, folder='.././training_results', trainingcycle=1, episode=1, state_size=5,
                 size=6000, save=True):
    # Plot newest state representation by SRL network
    closeAll()

    states_pca_2 = getPCA(np.array(states), 2)
    positions = np.array(data_dick["position_obs"])[-len(states):]
    rewards = np.array(rews)  # [-len(states):]
    timesteps = np.array(range(len(positions)))

    try:
        plot_2d_colormap(xy=positions, val=timesteps,
                         name="Agent position measurement estimated by odom during training 1", label="Timesteps",
                         color="YlGn", plotgoal=True, clip_val=False)
        plot_2d_colormap(xy=positions, val=positions[:, 2],
                         name="Agent position measurement estimated by odom during training", label="Angle",
                         color="YlGn", plotgoal=True, clip_val=False)
    except:
        print('Could not plot ground truth data')

    plot_2d_colormap(xy=np.array(data_dick["position_obs"]), val=np.array(data_dick["rewards"], dtype=np.float32),
                     name="Agent position versus reward", label="Reward",
                     plotgoal=False, clip_val=True)

    try:
        plot_2d_colormap(xy=np.array(data_dick["position_obs"]), val=np.array(data_dick["qvalue"], dtype=np.float32),
                         name="Agent position versus Q value", label="Q value",
                         plotgoal=False, clip_val=False)
    except:
        print('No q-value found in the data dictionary')

    plot_2d_colormap(xy=states_pca_2[-size:], val=rewards[-size:],
                     name="First two principal components after training",
                     label="Reward", clip_val=True)

    pos_pca = getPCA(np.array(positions)[-size:], 2)
    plot_2d_colormap(xy=pos_pca, val=rewards[-size:],
                     name="First two principal components of ground truth states",
                     label="Reward", clip_val=True)

    states_pca_3 = getPCA(np.array(states), 3)
    plot_3d_colormap(xyz=states_pca_3[-size:], val=rewards[-size:],
                     name="First three principal components after training", clip_val=True)

    # plot state dimension 1 and 2
    plot_2d_colormap(xy=np.array(states[-size:]), val=rewards[-size:],
                     name="First two state dimensions after training", label="Reward", clip_val=True)

    # compute and plot PCA
    pca_5 = PCA(n_components=state_size)
    pca_5.fit(states)
    var_explained = pca_5.explained_variance_ratio_ * 100
    plot_bar(values=var_explained, name=" Covariance matrix eigenvalue analysis ", xaxis="PCA component",
             yaxis="Percentage of variance explained")

    if 'episode_reward' in data_dick.keys():
        # plot cumulated reward
        ma_cumulative_reward = moving_average(np.asarray(data_dick["episode_reward"]), 10)
        plot_history(ma_cumulative_reward, name="Moving average reward history (10)", yaxis="Reward")

    # compute and plot TSNE
    try:
        states_embedded = TSNE(n_components=2, n_jobs=-1).fit_transform(np.array(states[-size:]))
        plot_2d_colormap(xy=states_embedded, val=rewards[-size:], name="TSNE visualization",
                         label="Reward", clip_val=True)
    except:
        print('could not make the TSNE plot')

    if save:
        saveMultipage(os.path.join(folder, "trainingdata", str(trainingcycle) + "_" + str(episode) + ".pdf"))
