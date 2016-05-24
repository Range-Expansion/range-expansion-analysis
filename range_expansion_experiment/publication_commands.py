__author__ = 'bryan'

import glob
import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ternary as ter
import pandas as pd
import image_analysis as rei
import scipy as sp
import scikits.bootstrap as bs


def bootstrap_column(gb, column_name, alpha=0.32):
    "Alpha is 1 - the confidence interval desired."
    data_list = []
    for name, cur_data in gb:
        data_list.append(cur_data[column_name].values)
    data_list = np.array(data_list)
    low_high = bs.ci(data_list, statfunction=lambda x: np.mean(x, axis=0), output='lowhigh', alpha=alpha)
    return low_high

def import_files_in_folder(title, quantity, i=None, j=None, base_directory='./'):
    folder_name = base_directory + title + '_' + quantity
    if (i is not None) and (j is not None):
        folder_name += '_' + str(i) + '_' + str(j)
    folder_name += '/'

    data_list = []

    files_to_import = glob.glob(folder_name + '*.pkl')
    for file_path in files_to_import:
        with open(file_path, 'rb') as fi:
            data_list.append(pkl.load(fi))

    # Sort files by radius

    radii = [z['r'] for z in data_list]
    order = np.argsort(radii)
    radii = [radii[z] for z in order]
    data_list = [data_list[z] for z in order]

    return data_list

def get_Fij_at_each_r(title, num_colors, base_directory='./'):
    Fij_dict_list = {}
    for i in range(num_colors):
        for j in range(i, num_colors):
            if i != j:
                Fij_dict_list[i, j] = import_files_in_folder(title, 'Fij_sym', i=i, j=j, base_directory=base_directory)
            else:
                Fij_dict_list[i, j] = import_files_in_folder(title, 'Fij', i=i, j=j, base_directory=base_directory)

    # We now want to organize each of these by radius. UGH.
    # Actually, each thing is sorted by radius. Sooooooo, assuming we do the same
    # radii for each, this isn't too bad.

    num_radii = len(Fij_dict_list[0, 0])

    Fij_at_each_r = []
    for r_index in range(num_radii):
        current_radius_Fij = {}
        for i in range(num_colors):
            for j in range(i, num_colors):
                current_radius_Fij[i, j] = Fij_dict_list[i, j][r_index]
        Fij_at_each_r.append(current_radius_Fij)

    return Fij_at_each_r

def make_x_axis_radians(unit=0.25, pi_range=np.pi, num_digits=2):
    const = pi_range / np.pi
    x_tick = np.arange(-const, const+unit, unit)

    x_label = [r"$" + format(r, '.'+str(num_digits)+'g')+ r"\pi$" for r in x_tick]
    ax = plt.gca()
    ax.set_xticks(x_tick*np.pi)
    ax.set_xticklabels(x_label)

def make_ternary_plot(input_fracs, label_list, color_list,  r_min=3.5, r_max=10, num_bins=200, offset_list=None,
                      cbar_ticks=None, ax = None, plot_cbar=True, plot_legend=True, alpha=0.8, textfontsize=20):
    if offset_list is None:
        offset_list = [[-15, -7], [-30, 20], [-30, -7]]

    if ax is None:
        fig, ax = ter.figure()
        #fig.set_size_inches(16, 10)
    else:
        fig = None

    ax.boundary(color='black', alpha=0.4)
    ax.gridlines(color='black', multiple=0.1)

    fracs_rebinned, bins = group_fracs(input_fracs, min_radius=r_min, max_radius=r_max, num_bins=num_bins, average_data=False)

    groups = fracs_rebinned.groupby('im_set')

    count = 0
    for group_name, cur_data in groups:
        ### Rebin ####

        fracs = cur_data.loc[:, 'ch0':'ch2'].values

        n_colors = fracs.shape[0]
        cmap = sns.cubehelix_palette(as_cmap=True, n_colors=n_colors, light=0.7, dark=.05)

        ax.plot_colored_trajectory(fracs, cmap, alpha=alpha, zorder=100)
        start_label = None
        finish_label = None
        if (count == len(groups) - 1):
            start_label = 'Start'
            finish_label = 'Finish'

        ax.scatter([fracs[0, :]], s=20, color='red', zorder=99999, marker='o', label=start_label)
        ax.scatter([fracs[-1, :]], s=20, color='blue', zorder=99999, marker='o', label=finish_label)

        count += 1

    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    plt.xlim(-.1, 1.1)
    plt.ylim(-.035, .95)

    new_ax = plt.gca()

    # Make a colorbar
    if plot_cbar:
        if cbar_ticks is None:
            ter.heatmapping.colorbar_hack(new_ax, r_min, r_max, cmap, text_format='%.2f',
                                         numticks=6, title='Radius (mm)')
        else:
            cbar = ter.heatmapping.colorbar_hack(new_ax, r_min, r_max, cmap, text_format='%.2f',
                                         numticks=6, title='Radius (mm)')
            cbar.set_ticks(cbar_ticks)

    ### Create annotations specifying the edges ####

    positions_list = [ter.project_point([1, 0, 0]), ter.project_point([0, 1, 0]),
                 ter.project_point([0, 0, 1])]

    for pos, label, offset, color in zip(positions_list, label_list, offset_list, color_list):
        new_ax.annotate(label, xy=pos, xytext=offset, ha='left',
                        va='top', xycoords='data',
                        textcoords='offset points',
                        fontsize=textfontsize, color=color)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    if plot_legend:
        plt.legend(loc='best')

    return ax

slicer = pd.IndexSlice
def make_mean_ternary_plot(input_fracs, label_list, color_list,  r_min=3.5, r_max=10, num_bins=200, offset_list=None,
                           plot_cbar=True, plot_legend=True, ax=None, textfontsize=20, linewidth=1.):
    if offset_list is None:
        offset_list = [[-15, -7], [-30, 20], [-30, -7]]

    if ax is None:
        fig, ax = ter.figure()
        #fig.set_size_inches(16, 10)
    else:
        fig = None

    ax.boundary(color='black', alpha=0.4)
    ax.gridlines(color='black', multiple=0.1, linewidth=linewidth)

    cur_data, bins = group_fracs(input_fracs, min_radius=r_min, max_radius=r_max, num_bins=num_bins)
    fracs = cur_data.loc[:, slicer['ch0':'ch2', 'mean']].values

    n_colors = fracs.shape[0]
    cmap = sns.cubehelix_palette(as_cmap=True, n_colors=n_colors, light=0.7, dark=.05)

    ax.plot_colored_trajectory(fracs, cmap, alpha=1, zorder=100)

    start_label = 'Start'
    finish_label = 'Finish'
    ax.scatter([fracs[0, :]], s=20, color='red', zorder=99999, marker='o', label=start_label)
    ax.scatter([fracs[-1, :]], s=20, color='blue', zorder=99999, marker='o', label=finish_label)

    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    plt.xlim(-.1, 1.1)
    plt.ylim(-.035, .95)

    new_ax = plt.gca()

    # Make a colorbar
    if plot_cbar:
        ter.heatmapping.colorbar_hack(new_ax, r_min, r_max, cmap, text_format='%.2f',
                                     numticks=6, title='Radius (mm)')

    ### Create annotations specifying the edges ####

    positions_list = [ter.project_point([1, 0, 0]), ter.project_point([0, 1, 0]),
                 ter.project_point([0, 0, 1])]

    for pos, label, offset, color in zip(positions_list, label_list, offset_list, color_list):
        new_ax.annotate(label, xy=pos, xytext=offset, ha='left',
                        va='top', xycoords='data',
                        textcoords='offset points',
                        fontsize=textfontsize, color=color)

    plt.gca().set_aspect('equal', adjustable='box')

    if plot_legend:
        plt.legend(loc='best')

    return ax

def group_fracs(fracs, min_radius=2.5, max_radius=10., num_bins=200, average_data=True):

    bins = np.linspace(min_radius, max_radius, num_bins)
    cut = pd.cut(fracs['radius_midbin_scaled'], bins)

    # Group each channel separately, first, take the mean...basically
    # get the same binning on each.

    imset_gb = fracs.groupby(['im_set', cut])
    rebinned_fracs = imset_gb.agg('mean')
    # Drop the problem column...
    rebinned_fracs.drop('radius_midbin_scaled', 1, inplace=True)
    rebinned_fracs = rebinned_fracs.reset_index()

    if not average_data:
        return rebinned_fracs, bins
    else:
        # Now that everything has the same binning, average.
        groups = rebinned_fracs.groupby('radius_midbin_scaled')
        result = groups.agg([np.mean, sp.stats.sem, np.var, np.std])
        result['radius_midbin_scaled'] = (bins[:-1] + bins[1:])/2.
        if result[result.isnull().any(axis=1)].values.shape[0] != 0:
            print 'Binning too tight, getting NaNs.'
        return result, bins


def make_twocolor_walk_plot(input_fracs, labels, colors, min_radius=3.5, max_radius=10, num_bins=150,
                            plot_mean=True):

    # Get average data
    mean_fracs, bins = group_fracs(input_fracs, min_radius=min_radius, max_radius=max_radius, num_bins=num_bins)
    midbins = (bins[:-1] + bins[1:])/2.

    plt.hold(True)

    # Loop over input fractions
    trajectories, bins = group_fracs(input_fracs, average_data=False, min_radius=min_radius, max_radius=max_radius, num_bins=num_bins)
    input_fracs = trajectories.groupby('im_set')

    num_imsets = len(input_fracs)
    new_cmap = sns.color_palette('husl', n_colors=num_imsets)

    count = 0
    for im_set, data in input_fracs:
        plt.plot(midbins, data['ch1'],
                color=new_cmap[count], linestyle='-', label='')
        count += 1

    if plot_mean:
        plt.plot(mean_fracs['radius_midbin_scaled'], mean_fracs['ch1', 'mean'],
                color='black', linestyle='--', label='Mean Trajectory')
        plt.legend(loc='best')

        # Plot the error
        y = mean_fracs['ch1', 'mean']
        yerr = mean_fracs['ch1', 'sem']
        x = mean_fracs['radius_midbin_scaled']
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2, color='black')

    plt.hold(False)
    plt.ylim(0, 1)
    plt.xlim(min_radius, max_radius) # Flips axis in y
    sns.despine(left=False, bottom=False, top=False, right=True)

    plt.xlabel('Radius (mm)')
    plt.ylabel('Fraction of eCFP')
    cur_ax = plt.gca()

    # Now label eCFP & eYFP

    cur_ax.annotate(labels[0], xy=(0, 1), xytext=(10, -10), ha='left', va='top',
                   xycoords='axes fraction', textcoords='offset points',
                   fontsize=20, color=colors[0])
    cur_ax.annotate(labels[1], xy=(0, 0), xytext=(10, 20), ha='left', va='top',
                   xycoords='axes fraction', textcoords='offset points',
                   fontsize=20, color=colors[1])

    # # Adding an arrow to the bottom axis...painful
    dps = plt.gcf().dpi_scale_trans.inverted()
    bbox = plt.gca().get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    delta_epsilon = .0003
    ymin= -delta_epsilon
    ymax= 1 - delta_epsilon
    xmin=max_radius
    xmax=max_radius + .2


    # manual arrowhead width and length
    hw = .2*(ymax-ymin)
    hl = .1*(xmax-xmin)
    lw = .5# axis line width
    ohg = 0.2 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    cur_ax.spines['left'].set_color('black')
    cur_ax.spines['top'].set_color('black')
    cur_ax.spines['right'].set_color('black')
    cur_ax.spines['bottom'].set_color('black')

    cur_ax.arrow(xmin, ymax, xmax-xmin, 0, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, overhang = ohg,
             length_includes_head= True, clip_on = False, color='black')

    cur_ax.arrow(xmin, ymin, xmax-xmin, 0, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, overhang = ohg,
             length_includes_head= True, clip_on = False, color='black')

    return cur_ax

