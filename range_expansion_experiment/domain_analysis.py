__author__ = 'bryan'

import glob
import cPickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ternary as ter
import pandas as pd
import image_analysis as rei


#### Domain Analysis #####

def get_cdf_quantity_domains(domains, quantity, num_channels, cdf_bins, filter_value=None):
    radius_bins = domains['radius_scaled_used']

    if filter_value is None:
        if quantity == 'lengths_scaled':
            filter_value = 0.3 # This is about the overlap region size
        elif quantity == 'angular_width':
            filter_value = 0.5 * (np.pi/180.)  # in pixels; 1 degree

    df_dict = {}
    for ch in range(num_channels):
        for radius_index in range(len(radius_bins)):
            cdf_list = []
            cur_data = domains[ch, radius_index]
            for imset_index, image_df in cur_data.groupby('imset_index'):
                if quantity == 'lengths_scaled':
                    image_df = image_df.loc[image_df['lengths_scaled'] >= filter_value, :]
                elif quantity == 'angular_width':
                    image_df = image_df.loc[image_df['angular_width'] >= filter_value, :]
                cumulative_counts = rei.Range_Expansion_Experiment.get_cumsum_quantity(image_df, cdf_bins,
                                                                                       quantity=quantity)
                cumulative_counts['imset_index'] = imset_index
                cumulative_counts['ecdf'] = cumulative_counts['cumsum'] / cumulative_counts['cumsum'].max()
                cdf_list.append(cumulative_counts)
            cdf_df = pd.concat(cdf_list)
            df_dict[ch, radius_index] =  cdf_df

    df_dict['radius_scaled_used'] = radius_bins
    return df_dict

def get_mean_domain_quantity(domains, quantity, num_channels, filter_value=None):
    radius_bins = domains['radius_scaled_used']

    if filter_value is None:
        if quantity == 'lengths_scaled':
            filter_value = 0.3 # This is about the overlap region size
        elif quantity == 'angular_width':
            filter_value = 1. * (np.pi/180.)  # in pixels; 1 degree

    df_dict = {}
    for ch in range(num_channels):
        r_list = []
        for radius_index in range(len(radius_bins)):
            mean_list = []
            cur_data = domains[ch, radius_index]
            for imset_index, image_df in cur_data.groupby('imset_index'):
                if quantity == 'lengths_scaled':
                    image_df = image_df.loc[image_df['lengths_scaled'] >= filter_value, :]
                elif quantity == 'angular_width':
                    image_df = image_df.loc[image_df['angular_width'] >= filter_value, :]
                mean_quantity = image_df[quantity].mean()
                mean_list.append(mean_quantity)
            mean_df = pd.DataFrame({quantity + '_mean' : mean_list})
            mean_df['radius_scaled'] = radius_bins[radius_index]
            r_list.append(mean_df)
        r_df = pd.concat(r_list)
        df_dict[ch] = r_df
    return df_dict

def get_number_of_domains_per_channel(domains, num_channels, filter_value=None):
    radius_bins = domains['radius_scaled_used']

    if filter_value is None:
        filter_value = 0.3 # This is about the overlap region size

    df_dict = {}
    for ch in range(num_channels):
        r_list = []
        for radius_index in range(len(radius_bins)):
            num_domain_list = []
            cur_data = domains[ch, radius_index]
            for imset_index, image_df in cur_data.groupby('imset_index'):
                image_df = image_df.loc[image_df['lengths_scaled'] >= filter_value, :]
                num_domains = image_df.shape[0]
                num_domain_list.append(num_domains)
            mean_df = pd.DataFrame({'num_domains' : num_domain_list})
            mean_df['radius_scaled'] = radius_bins[radius_index]
            r_list.append(mean_df)
        r_df = pd.concat(r_list)
        df_dict[ch] = r_df
    return df_dict

def get_total_number_of_domains(domains, num_channels, filter_value=None):
    radius_bins = domains['radius_scaled_used']

    if filter_value is None:
        filter_value = 0.3 # This is about the overlap region size

    r_list = []
    for radius_index in range(len(radius_bins)):
        cur_data_list = []
        for ch in range(num_channels):
            cur_data = domains[ch, radius_index]
            cur_data_list.append(cur_data)

        cur_data = pd.concat(cur_data_list)
        num_domain_list = []
        for imset_index, image_df in cur_data.groupby('imset_index'):
            image_df = image_df.loc[image_df['lengths_scaled'] >= filter_value, :]
            num_domains = image_df.shape[0]
            num_domain_list.append(num_domains)
        total_df = pd.DataFrame({'num_domains' : num_domain_list})
        total_df['radius_scaled'] = radius_bins[radius_index]
        r_list.append(total_df)
    r_df = pd.concat(r_list)

    return r_df

def plot_mean_quantity(mean_dict, ch, quantity, r_min=3.5, **kwargs):
    cur_data = mean_dict[ch]
    cur_data = cur_data.groupby('radius_scaled').agg('mean')
    cur_data = cur_data.reset_index()

    cur_data = cur_data.loc[cur_data['radius_scaled'] >= r_min, :]

    plt.plot(cur_data['radius_scaled'], cur_data[quantity + '_mean'], **kwargs)

def plot_mean_num_domains(num_domains, ch, r_min=3.5, **kwargs):
    cur_data = num_domains[ch]
    cur_data = cur_data.groupby('radius_scaled').agg('mean')
    cur_data = cur_data.reset_index()

    cur_data = cur_data.loc[cur_data['radius_scaled'] >= r_min, :]

    plt.plot(cur_data['radius_scaled'], cur_data['num_domains'], **kwargs)

def plot_average_total_num_domains(total_num_domains, r_min=3.5, **kwargs):

    cur_data = total_num_domains.groupby('radius_scaled').agg('mean')
    cur_data = cur_data.reset_index()

    cur_data = cur_data.loc[cur_data['radius_scaled'] >= r_min, :]

    plt.plot(cur_data['radius_scaled'], cur_data['num_domains'], **kwargs)

def plot_domain_cdf(cdf_dict, cur_channel, quantity, min_radius=3.5, use_legend=True, plot_every=0.5):

    desired_r_bins = np.arange(min_radius, 10 + plot_every, plot_every)
    r_bins = np.array(cdf_dict['radius_scaled_used'])
    r_bins = np.around(r_bins, 3) # This way we can conveniently compare floats...

    # Figure out which r_bins are in the desired_r_bins
    r_bins_mask = np.in1d(r_bins, desired_r_bins)

    r_bins_filtered = r_bins[r_bins_mask]

    colors_to_use = sns.cubehelix_palette(n_colors = len(r_bins_filtered))
    count = 0
    plt.hold(True)
    for i, plot_this_value in zip(range(len(r_bins)), r_bins_mask): # Number of bins
        if plot_this_value:
            cur_data = cdf_dict[cur_channel, i]
            mean_data = cur_data.groupby(level=0).agg('mean')
            plt.plot(mean_data[quantity + '_midbin'], mean_data['ecdf'],
                    color=colors_to_use[count], label=r_bins[i])
            count += 1
    plt.hold(False)

    if use_legend:
        plt.legend(loc='best')
    plt.ylabel('Average Empirical CDF')

def plot_domain_cdf_trajectories(cdf_dict, cur_channel, quantity, min_radius=3.5, xlim=None):
    r_bins = cdf_dict['radius_scaled_used']

    for i in range(len(r_bins)): # Number of bins
        cur_r = r_bins[i]
        if cur_r >= min_radius:
            plt.figure()
            plt.hold(True)
            cur_data = cdf_dict[cur_channel, i]
            # Group by imset
            for im_set, image_df in cur_data.groupby('imset_index'):
                plt.plot(image_df[quantity + '_midbin'], image_df['ecdf'])
            plt.title(r_bins[i])
            plt.hold(False)

            if xlim is not None:
                plt.xlim(xlim[0], xlim[1])

def plot_all_domain_quantities(domains, num_colors, label_list, color_list):
    mean_length_dict = get_mean_domain_quantity(domains, 'lengths_scaled', num_colors)
    plt.figure()
    plt.hold(True)
    for ch in range(num_colors):
        plot_mean_quantity(mean_length_dict, ch, 'lengths_scaled', marker='.', color=color_list[ch], label=label_list[ch])
        plt.xlabel('Radius (mm)')
        plt.ylabel('Average Domain Size (mm)')
    plt.hold(False)
    plt.legend(loc='best')

    mean_angular_dict = get_mean_domain_quantity(domains, 'angular_width', num_colors)
    plt.figure()
    plt.hold(True)
    for ch in range(num_colors):
        plot_mean_quantity(mean_angular_dict, ch, 'angular_width', marker='.', color=color_list[ch], label=label_list[ch])
        plt.xlabel('Radius (mm)')
        plt.ylabel('Average Domain Angular Width')
    plt.hold(False)
    plt.legend(loc='best')

    num_domains = get_number_of_domains_per_channel(domains, num_colors)
    plt.figure()
    plt.hold(True)
    for ch in range(num_colors):
        plot_mean_num_domains(num_domains, ch, marker='.', color=color_list[ch], label=label_list[ch])
        plt.xlabel('Radius (mm)')
        plt.ylabel('Number of Domains')
    plt.hold(False)
    plt.legend(loc='best')

    total_num_domains = get_total_number_of_domains(domains, num_colors)
    plt.figure()
    plot_average_total_num_domains(total_num_domains, marker='.', linestyle='-')
    plt.xlabel('Radius (mm)')
    plt.ylabel('Average Total Number of Domains')

    length_bins = np.linspace(0, 30, 800)
    cdf_lengths_dict = get_cdf_quantity_domains(domains, 'lengths_scaled', num_colors, length_bins)
    for ch in range(num_colors):
        plt.figure()
        plot_domain_cdf(cdf_lengths_dict, ch, 'lengths_scaled', plot_every=0.5)
        plt.xlabel('Domain Length (mm)')

    angular_bins = np.linspace(0, 2*np.pi, 800)
    cdf_angular_dict = get_cdf_quantity_domains(domains, 'angular_width', num_colors, angular_bins)
    for ch in range(num_colors):
        plt.figure()
        plot_domain_cdf(cdf_angular_dict, ch, 'angular_width', plot_every=0.5)
        plt.xlabel('Angular Width')
        plt.xlim(0, np.pi/2)