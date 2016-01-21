__author__ = 'bryan'
#skimage imports
import skimage as ski
import skimage.io
import skimage.measure
import skimage.morphology
#other stuff
import matplotlib.pyplot as plt
import tifffile as ti
import xml.etree.ElementTree as ET
import glob
import os
import pandas as pd
import numpy as np
import scipy as sp
import mahotas as mh
import cPickle as pkl
import gc
import seaborn as sns

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    three 1d arrays:  start indicies, stop indicies and lengths of contigous regions.
    """

    d = np.diff(condition)
    idx, = d.nonzero()
    idx += 1 # need to shift indices because of diff

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    starts = idx[0::2]
    stops = idx[1::2]
    lengths = stops - starts

    return starts, stops, lengths

class Publication_Experiment(object):
    """Choose whether you have enough memory to store images in RAM or not. If you do, things go much faster."""

    def __init__(self, experiment_path, cache=False, title = None, annih_min_radius=3.5, data_export_directory='./', **kwargs):
        """Only one experiment per class now."""
        self.experiment_path = experiment_path
        self.title = title
        self.cache = cache

        self.hetero_r_list = [2.5, 3, 3.5, 4, 5, 6, 8, 10] # Radii used to compare heterozygosity
        self.num_theta_bins_list = [500, 600, 700, 800, 1000, 1000, 1500, 1500] # Bins at each radius; larger radii allow more bins
        self.domain_r_bins = np.arange(3.5, 10.1, .1)

        self.experiment = None # Used to point to a current experiment
        self.complete_masks = None
        self.complete_annih = None

        # Initialize the experiment
        self.experiment = Range_Expansion_Experiment(self.experiment_path, title=self.title, cache=self.cache, **kwargs)
        self.complete_radii = self.experiment.get_complete_im_sets('circle_folder')
        self.complete_masks = self.experiment.get_complete_im_sets('masks_folder')
        self.complete_annih = self.experiment.get_complete_im_sets('annihilation_folder')

        self.annih_min_radius = annih_min_radius

        self.data_export_directory = data_export_directory

    def write_nonlocal_quantity_to_disk(self, quantity_str, i= None, j=None):
        # If caching, initialize the experiment once and go from there.

        if self.cache:
            for q in self.complete_masks:
                    self.experiment.image_set_list[q].finish_setup()

        for r, num_theta_bins in zip(self.hetero_r_list, self.num_theta_bins_list):
            print r

            if (i is None) and (j is None):
                folder_name = self.data_export_directory + self.experiment.title + '_' + quantity_str
            else:
                folder_name = self.data_export_directory +  self.experiment.title + '_' + quantity_str + '_' + str(i) + '_' + str(j)

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            quantity_info = {}

            quantity_info['r'] = r
            quantity_info['num_theta_bins'] = num_theta_bins

            quantity = None

            # Determine if you need to clear memory and initialize...depends on if you want to cache
            if self.cache:
                initialize_and_clear_memory = False
            else:
                initialize_and_clear_memory = True

            if quantity_str == 'hetero':
                quantity = self.experiment.get_nonlocal_hetero_averaged(self.complete_masks, r, num_theta_bins=num_theta_bins,
                                                            skip_grouping=True, calculate_overlap=True,
                                                            initialize_and_clear_memory=initialize_and_clear_memory)
            elif quantity_str == 'Fij_sym':
                quantity = self.experiment.get_nonlocal_Fij_sym_averaged(self.complete_masks, i, j, r, num_theta_bins=num_theta_bins,
                                                            skip_grouping=True, calculate_overlap=True,
                                                            initialize_and_clear_memory = initialize_and_clear_memory)
            elif quantity_str == 'Ftot':
                quantity = self.experiment.get_nonlocal_Ftot_averaged(self.complete_masks, r, num_theta_bins=num_theta_bins,
                                                            skip_grouping=True, calculate_overlap=True,
                                                            initialize_and_clear_memory = initialize_and_clear_memory)
            elif quantity_str == 'Fij':
                quantity = self.experiment.get_nonlocal_Fij_averaged(self.complete_masks, i, j, r, num_theta_bins=num_theta_bins,
                                                            skip_grouping=True, calculate_overlap=True,
                                                            initialize_and_clear_memory = initialize_and_clear_memory)

            # Clear memory
            if not self.cache:
                del self.experiment

            quantity_info['quantity'] = quantity
            with open(folder_name + '/' + str(r) + '.pkl', 'wb') as fi:
                pkl.dump(quantity_info, fi)

            # Clear memory
            del quantity
            del quantity_info

            if not self.cache:
                self.experiment = Range_Expansion_Experiment(self.experiment_path, title=self.title, cache=self.cache,
                                                bigger_than_image=False) #TODO make bigger than image better behaved

            gc.collect()

    def write_annih_coal_to_disk(self, **kwargs):

        for q in self.complete_annih:
            self.experiment.image_set_list[q].finish_setup()

        combined_annih, combined_coal = self.experiment.get_cumulative_average_annih_coal(self.complete_annih,
                                                                                          min_radius_scaled=self.annih_min_radius,
                                                                                          **kwargs)
        with open(self.data_export_directory + self.experiment.title + '_annih.pkl', 'wb') as fi:
            pkl.dump(combined_annih, fi)
        with open(self.experiment.title + '_coal.pkl', 'wb') as fi:
            pkl.dump(combined_coal, fi)

    def write_annihilation_asymmetry_to_disk(self, **kwargs):

        for q in self.complete_annih:
            self.experiment.image_set_list[q].finish_setup()

        combined_deltaP = self.experiment.get_annihilation_asymmetry(self.complete_annih,
                                                                     min_radius_scaled=self.annih_min_radius,
                                                                     **kwargs)
        with open(self.data_export_directory + self.experiment.title + '_annih_asymmetry.pkl', 'wb') as fi:
            pkl.dump(combined_deltaP, fi)

    def write_fraction_trajectories_to_disk(self, min_radius=2.5, max_radius=10):
        "Don't worry about memory here."

        fracs = self.experiment.get_fractions_concat(self.complete_masks)
        fracs = fracs.loc[fracs['radius_midbin_scaled'] > min_radius, :]
        fracs = fracs.loc[fracs['radius_midbin_scaled'] < max_radius, :]

        # Write the information to disk
        folder_name = self.data_export_directory + self.experiment.title + '_fractions'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(folder_name + '/fractions.pkl', 'wb') as fi:
            pkl.dump(fracs, fi)

    def write_domain_sizes_to_disk(self, input_bins=None):
        if input_bins is None:
            input_bins = self.domain_r_bins
        domain_sizes = self.experiment.get_domain_sizes_at_radii(self.complete_masks, input_bins)

        # Write the information to disk
        folder_name = self.data_export_directory + self.experiment.title + '_domain_sizes'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(folder_name + '/domain_sizes.pkl', 'wb') as fi:
            pkl.dump(domain_sizes, fi)

    #### Summary Method ####

    def write_all_to_disk(self, num_colors):
        print 'Calculating & writing heterozygosity'
        self.write_nonlocal_quantity_to_disk('hetero')

        for i in range(num_colors):
            for j in range(i, num_colors):
                if i != j:
                    print 'Calculating & writing F' + str(i) + str(j) + '_sym'
                    self.write_nonlocal_quantity_to_disk('Fij_sym', i=i, j=j)
                elif i == j:
                    print 'Calculating & writing F' +str(i) + str(j)
                    self.write_nonlocal_quantity_to_disk('Fij', i=i, j=j)

        print 'Calculating & writing annihilations & coalescences'
        self.write_annih_coal_to_disk()
        print 'Calculating & writing trajectories'
        self.write_fraction_trajectories_to_disk()
        print 'Calculating & writing domains'
        self.write_domain_sizes_to_disk()

    #### Importing Methods #####
    def import_files_in_folder(self, quantity, i=None, j=None):
        folder_name = self.data_export_directory + self.title + '_' + quantity
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
        data_list = [data_list[z] for z in order]

        return data_list

    def get_Fij_at_each_r(self, num_colors):
        Fij_dict_list = {}
        for i in range(num_colors):
            for j in range(i, num_colors):
                if i != j:
                    Fij_dict_list[i, j] = self.import_files_in_folder('Fij_sym', i=i, j=j)
                else:
                    Fij_dict_list[i, j] = self.import_files_in_folder('Fij', i=i, j=j)

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

    #### Plotting Methods ####

    @staticmethod
    def plot_average_domain_cdf(domain_df, r_bins, channel_num):
        colors_to_use = sns.cubehelix_palette(n_colors=r_bins.shape[0])

        count = 0
        cur_channel = 0
        for i in range(r_bins.shape[0]): # Number of bins
            cur_data = domain_df[cur_channel, i]
            mean_data = cur_data.groupby(level=0).agg('mean')
            plt.plot(mean_data['lengths_scaled_midbin'], mean_data['ecdf'],
                    color=colors_to_use[count], label=r_bins[i])
            count += 1

        plt.title('Average Empirical CDF of Domain Sizes (ch' + str(channel_num) + ') vs. R')
        plt.legend(loc='best')
        plt.xlabel('Domain Size (mm)')
        plt.ylabel('Average Empirical CDF')

class Range_Expansion_Experiment(object):
    def __init__(self, base_folder, cache=True, title=None, **kwargs):
        """Cache determines whether data is cached; it can vastly speed up everything."""
        self.cache = cache
        self.title=title

        self.path_dict = {}
        self.path_dict['circle_folder'] = base_folder + 'circle_radius/'
        self.path_dict['edges_folder'] = base_folder + 'edges/'
        self.path_dict['doctored_edges_folder'] = base_folder + 'edges_doctored/'
        self.path_dict['masks_folder'] = base_folder + 'masks/'
        self.path_dict['tif_folder'] = base_folder + 'tif/'
        self.path_dict['annihilation_folder'] = base_folder + 'annihilation_and_coalescence/'
        self.path_dict['homeland_folder'] = base_folder + 'homeland/'

        self.path_dict['labeled_domains'] = base_folder + 'labeled_domains/'

        self.image_set_list = None

        self.finish_setup(**kwargs)

    def get_domain_dfs(self, indices_to_use, **kwargs):
        df_list = []
        for index in indices_to_use:
            cur_imset = self.image_set_list[index]
            df = cur_imset.get_domain_dfs(**kwargs)
            df['imset_index'] = index

            df_list.append(df)
        return pd.concat(df_list)

    def get_edge_dfs(self, indices_to_use, **kwargs):
        df_list = []
        for index in indices_to_use:
            cur_imset = self.image_set_list[index]
            df, midbins = cur_imset.get_edge_df(**kwargs)
            df['imset_index'] = index

            df_list.append(df)
        return pd.concat(df_list), midbins

    def finish_setup(self, **kwargs):
        # Get a list of all images
        tif_paths = glob.glob(self.path_dict['tif_folder'] + '*.ome.tif')
        tif_paths.sort()

        # Now setup each set of images
        image_names = [os.path.basename(z) for z in tif_paths]

        self.image_set_list = []

        for cur_name in image_names:
            im_set = Image_Set(cur_name, self.path_dict, cache=self.cache, **kwargs)
            self.image_set_list.append(im_set)

    # Convenience method
    def get_complete_im_sets(self, folder_name):
        complete_im_sets = []

        folder = self.path_dict[folder_name]
        desired_names = glob.glob(folder + '*.tif')
        # Get the basename of the desired_names
        desired_names = [os.path.basename(z) for z in desired_names]
        count = 0
        for cur_im_set in self.image_set_list:
            if cur_im_set.image_name in desired_names:
                complete_im_sets.append(count)
            count += 1
        return complete_im_sets

    ## Work with Averaging multiple sets of data

    def get_fractions_concat(self, im_set_indices_to_use):
        """Returns a DF with all fraction lists. Also returns the cut that should be used to groupby."""
        frac_list = []

        for index in im_set_indices_to_use:
            cur_im_set = self.image_set_list[index]
            fracs = cur_im_set.get_fracs_at_radius()
            fracs['im_set'] = index
            fracs['bio_replicate'] = cur_im_set.get_biorep_name()
            fracs.reset_index(inplace=True, drop=True)

            frac_list.append(fracs)

        frac_concat = pd.concat(frac_list)
        return frac_concat

    @staticmethod
    def get_cumsum_quantity(df, bins, quantity='radius_scaled'):
        """Calculates cumulative sums for annihilations/coalescences along bins which can later be averaged."""

        # Drop all unecessary columns...or this becomes extremely confusing
        df['count'] = 1
        df = df.loc[:, [quantity, 'count']]

        cut = pd.cut(df[quantity], bins)
        gb = df.groupby(cut)
        mean_bins = gb.agg('sum')
        mean_bins.loc[pd.isnull(mean_bins['count']), 'count'] = 0
        mean_bins['cumsum'] = np.cumsum(mean_bins['count'])

        # Add radius_midbin to each...
        mean_bins[quantity + '_midbin'] = (bins[:-1] + bins[1:])/2.

        # Drop the quantity, in this case lengths_scaled, to avoid confusion. You just
        # need the midbin...
        mean_bins = mean_bins.drop(quantity, axis=1)

        return mean_bins

    def get_cumulative_average_annih_coal(self, im_set_indices_to_use, min_radius_scaled = 2.5, max_radius_scaled = 11,
                                          num_bins = 500):
        new_r_bins = np.linspace(min_radius_scaled, max_radius_scaled, num_bins)
        binned_annih_list = []
        binned_coal_list = []
        for index in im_set_indices_to_use:
            cur_im_set = self.image_set_list[index]
            annih, coal = cur_im_set.get_annih_and_coal()

            binned_annih = Range_Expansion_Experiment.get_cumsum_quantity(annih, new_r_bins)
            binned_coal  = Range_Expansion_Experiment.get_cumsum_quantity(coal, new_r_bins)

            binned_annih['index'] = index
            binned_coal['index'] = index
            binned_annih['bio_replicate'] = cur_im_set.get_biorep_name()
            binned_coal['bio_replicate'] = cur_im_set.get_biorep_name()

            binned_annih_list.append(binned_annih)
            binned_coal_list.append(binned_coal)

        combined_annih = pd.concat(binned_annih_list, join='outer')
        combined_coal = pd.concat(binned_coal_list)

        return combined_annih, combined_coal

    def get_annihilation_asymmetry(self, im_set_indices_to_use, min_radius_scaled = 3.5, max_radius_scaled = 11,
                                          num_bins = 7):
        new_r_bins = np.linspace(min_radius_scaled, max_radius_scaled, num_bins)
        deltaP_list = []
        for index in im_set_indices_to_use:
            cur_im_set = self.image_set_list[index]
            annih, coal = cur_im_set.get_annih_and_coal()

            # index by the new r bins
            annih_cut = pd.cut(annih['radius_scaled'], new_r_bins)
            annih_levels = annih_cut.cat.categories
            annih_binned = pd.value_counts(annih_cut).reindex(annih_levels)

            coal_cut = pd.cut(coal['radius_scaled'], new_r_bins)
            coal_levels = coal_cut.cat.categories
            coal_binned = pd.value_counts(coal_cut).reindex(coal_levels)

            deltaP_array = (annih_binned - coal_binned)/(annih_binned + coal_binned)

            deltaP_df = deltaP_array.to_frame(name='deltaP')
            deltaP_df['radius_midbin_scaled'] = (new_r_bins[1:] + new_r_bins[0:-1])/2.

            deltaP_df['imset_index'] = index
            deltaP_df['bio_replicate'] = cur_im_set.get_biorep_name()
            deltaP_df['bio_replicate'] = cur_im_set.get_biorep_name()

            deltaP_list.append(deltaP_df)

        combined_list = pd.concat(deltaP_list)

        return combined_list

    @staticmethod
    def bin_multiple_df_on_r_getmean(df_list, max_r_scaled, num_r_bins = 600):
        # Set up binning
        rscaled_bins = np.linspace(0, max_r_scaled, num=num_r_bins)

        mean_list = []
        # Loop through im_sets, bin at each r
        for cur_df in df_list:
            # Assumes that the dataframes have a scaled radius column
            r_scaled_cut = pd.cut(cur_df['radius_scaled'], rscaled_bins)
            r_scaled_groups = cur_df.groupby(r_scaled_cut)
            cur_im_mean = r_scaled_groups.mean()
            # Check for nan's
            nan_list = cur_im_mean[cur_im_mean.isnull().any(axis=1)]
            if not nan_list.empty:
                print 'r binning is too tight; getting NaN'
                print nan_list
            # Append list
            mean_list.append(cur_im_mean),
        # Combine the list of each experiment
        return mean_list

    def get_local_hetero_averaged(self, im_sets_to_use, num_r_bins=800):
        '''Assumes that the images are already setup.'''

        # Get the maximum radius to bin, first of all
        max_r_scaled = 99999 # in mm; this is obviously ridiculous, nothing will be larger
        for im_set_index in im_sets_to_use:
            cur_im = self.image_set_list[im_set_index]
            cur_max_r = cur_im.max_radius * cur_im.get_scaling()
            if cur_max_r < max_r_scaled:
                max_r_scaled = cur_max_r

        # Set up binning
        rscaled_bins = np.linspace(0, max_r_scaled, num=num_r_bins)

        local_hetero_df_list = []
        # Loop through im_sets, bin at each r
        for im_set_index in im_sets_to_use:
            cur_im = self.image_set_list[im_set_index]
            local_hetero = cur_im.get_local_hetero_df()
            local_hetero_df_list.append(local_hetero)

        mean_list = self.bin_multiple_df_on_r_getmean(local_hetero_df_list, max_r_scaled, num_r_bins=num_r_bins)
        # Combine the list of each experiment
        combined_mean_df = pd.concat(mean_list)
        # Group by the index
        result = combined_mean_df.groupby(level=0, axis=0).agg(['mean', sp.stats.sem])
        # Sort by radius scaled
        result = result.sort([('radius_scaled', 'mean')])
        # Create a column with the midpoint of each bin which is what we actually want
        result['r_scaled_midbin'] = (rscaled_bins[1:] + rscaled_bins[0:-1])/2.

        return result

    def get_overlap_density_averaged(self, im_sets_to_use, num_overlap, num_r_bins=800):
        '''Assumes that the images are already setup.'''

        # Get the maximum radius to bin, first of all
        max_r_scaled = 99999 # in mm; this is obviously ridiculous, nothing will be larger
        for im_set_index in im_sets_to_use:
            cur_im = self.image_set_list[im_set_index]
            cur_max_r = cur_im.max_radius * cur_im.get_scaling()
            if cur_max_r < max_r_scaled:
                max_r_scaled = cur_max_r

        # Set up binning
        rscaled_bins = np.linspace(0, max_r_scaled, num=num_r_bins)

        overlap_df_list = []
        # Loop through im_sets, bin at each r
        for im_set_index in im_sets_to_use:
            cur_im = self.image_set_list[im_set_index]
            edge_df = cur_im.get_overlap_df(num_overlap)
            overlap_df_list.append(edge_df)

        mean_list = self.bin_multiple_df_on_r_getmean(overlap_df_list, max_r_scaled, num_r_bins=num_r_bins)
        # Combine the list of each experiment
        combined_mean_df = pd.concat(mean_list)
        # Group by the index
        result = combined_mean_df.groupby(level=0, axis=0).agg(['mean', sp.stats.sem])
        # Sort by radius scaled
        result = result.sort([('radius_scaled', 'mean')])
        # Create a column with the midpoint of each bin which is what we actually want
        result['r_scaled_midbin'] = (rscaled_bins[1:] + rscaled_bins[0:-1])/2.

        return result

    def get_black_corrected_average_frac(self, im_sets_to_use, min_r = 2.5, max_r = 10, num_bins = 300):

        num_channels = self.image_set_list[im_sets_to_use[0]].fluorescent_mask.shape[0]

        r_scaled = np.linspace(min_r, max_r, num_bins)
        df_list = []
        for i in im_sets_to_use:
            cur_im_set = self.image_set_list[i]

            cur_scaling = cur_im_set.get_scaling()
            r_pixel = np.around(r_scaled / cur_scaling)
            fracs = cur_im_set.get_average_fractions_black_corrected(r_pixel)
            frac_dict = {}
            for ch_num in range(num_channels):
                frac_dict['ch' + str(ch_num)] = fracs[:, ch_num]
            frac_dict['imset_index'] = i

            frac_dict['radius_scaled'] = r_scaled

            cur_df = pd.DataFrame(frac_dict)
            df_list.append(cur_df)
        df_list = pd.concat(df_list)

        return df_list

    def get_domain_sizes_at_radii(self, im_sets_to_use, r_scaled_bins):
        """Returns a dictionary with keys dict[channel_num, radius_index (i.e. 12th bin in the radius index)"""

        df_list = []
        for i in im_sets_to_use:
            cur_im_set = self.image_set_list[i]

            cur_scaling = cur_im_set.get_scaling()
            r_index_count = 0
            for cur_r in r_scaled_bins:
                r_pixel = np.around(cur_r / cur_scaling)
                domains_df = cur_im_set.get_domain_sizes_scaled_at_radius(r_pixel)

                domains_df['imset_index'] = i
                domains_df['radius_scaled'] = cur_r
                domains_df['r_index_count'] = r_index_count
                r_index_count += 1

                df_list.append(domains_df)
        domain_sizes = pd.concat(df_list)

        # Now organize this data.

        data_dict = {}

        for cur_channel, channel_data in domain_sizes.groupby('channel'):
            for cur_rindex, data_at_r in channel_data.groupby('r_index_count'):
                data_dict[cur_channel, cur_rindex] = data_at_r
        data_dict['radius_scaled_used'] = r_scaled_bins
        return data_dict

    def get_nonlocal_quantity_averaged(self, nonlocal_quantity, im_sets_to_use, r_scaled, num_theta_bins=250, delta_x=1.5,
                                       skip_grouping=False, calculate_overlap=False,
                                       initialize_and_clear_memory=False, **kwargs):
        df_list = []
        standard_theta_bins = np.linspace(-np.pi, np.pi, num_theta_bins)
        midbins = (standard_theta_bins[1:] + standard_theta_bins[0:-1])/2.

        for im_set_index in im_sets_to_use:
            cur_im_set = self.image_set_list[im_set_index]
            if initialize_and_clear_memory:
                cur_im_set.finish_setup()
            cur_scaling = cur_im_set.get_scaling()

            desired_r = np.around(r_scaled / cur_scaling)
            returned_dict, quantity_shortname = None, None
            if nonlocal_quantity == 'hetero':
                returned_dict = cur_im_set.get_nonlocal_hetero(desired_r, delta_x = delta_x,
                                                               calculate_overlap=calculate_overlap, **kwargs)
                quantity_shortname = 'h'
            elif nonlocal_quantity == 'Ftot':
                returned_dict = cur_im_set.get_nonlocal_Ftot(desired_r, delta_x=delta_x,
                                                             calculate_overlap=calculate_overlap, **kwargs)
                quantity_shortname = 'Ftot'
            elif nonlocal_quantity == 'Fij':
                returned_dict = cur_im_set.get_nonlocal_Fij(desired_r, delta_x=delta_x,
                                                            calculate_overlap=calculate_overlap, **kwargs)
                quantity_shortname = 'Fij'
            elif nonlocal_quantity == 'Fij_sym':
                returned_dict = cur_im_set.get_nonlocal_Fij_sym(desired_r, delta_x=delta_x,
                                                                calculate_overlap=calculate_overlap, **kwargs)
                quantity_shortname = 'Fij_sym'

            if initialize_and_clear_memory:
                cur_im_set.unitialize()

            result = returned_dict['result']
            theta_list = returned_dict['theta_list']
            overlap = returned_dict['average_overlap']

            # Create the df
            if overlap is None:
                cur_df = pd.DataFrame({quantity_shortname: result, 'theta': theta_list})
            else:
                cur_df = pd.DataFrame({quantity_shortname: result, 'theta': theta_list, 'overlap': overlap})

            # Check for nan's caused by heterozygosity
            if not cur_df[cur_df.isnull().any(axis=1)].empty:
                print 'Nonlocal quantity returning nan rows from im_set=' + str(im_set_index) + ', r=' + str(r_scaled)
                print cur_df[cur_df.isnull().any(axis=1)]

            cur_df['radius_scaled'] = r_scaled
            cur_df['im_set_index'] = im_set_index

            # Now bin on the standard theta bins
            theta_cut = pd.cut(cur_df['theta'], standard_theta_bins)
            cur_df = cur_df.groupby(theta_cut).mean()
            cur_df['theta_midbin'] = midbins

            df_list.append(cur_df)

        # Now combine the df_list
        combined_df = pd.concat(df_list)
        if not skip_grouping:
            # Groupby the index which is theta
            groups = combined_df.groupby(level=0, axis=0)
            # Get statistics
            combined_df = groups.agg(['mean', sp.stats.sem])
            # Sort the results by theta
            combined_df = combined_df.sort([('theta', 'mean')])

        # Check for nan's due to binning
        if not combined_df[combined_df.isnull().any(axis=1)].empty:
            print 'Binning is too tight; getting NaN at r=' + str(r_scaled)
            print combined_df[combined_df.isnull().any(axis=1)]

        return combined_df

    def get_nonlocal_hetero_averaged(self, im_sets_to_use, r_scaled, num_theta_bins=500, delta_x=1.5, **kwargs):
        return self.get_nonlocal_quantity_averaged('hetero', im_sets_to_use, r_scaled, num_theta_bins=num_theta_bins,
                                                   delta_x=delta_x, **kwargs)

    def get_nonlocal_Ftot_averaged(self, im_sets_to_use, r_scaled, num_theta_bins=500, delta_x=1.5, **kwargs):
        return self.get_nonlocal_quantity_averaged('Ftot', im_sets_to_use, r_scaled, num_theta_bins=num_theta_bins,
                                                   delta_x=delta_x, **kwargs)

    def get_nonlocal_Fij_averaged(self, im_sets_to_use, i, j, r_scaled, num_theta_bins=500, delta_x=1.5, **kwargs):
        return self.get_nonlocal_quantity_averaged('Fij', im_sets_to_use, r_scaled, i=i, j=j,
                                                   num_theta_bins=num_theta_bins, delta_x=delta_x, **kwargs)

    def get_nonlocal_Fij_sym_averaged(self, im_sets_to_use, i, j, r_scaled, num_theta_bins=500, delta_x=1.5, **kwargs):
        return self.get_nonlocal_quantity_averaged('Fij_sym', im_sets_to_use, r_scaled, i=i, j=j,
                                                   num_theta_bins=num_theta_bins, delta_x=delta_x, **kwargs)

class Image_Set(object):
    '''Homeland radius is used to get the center of the expansion now.'''
    def __init__(self, image_name, path_dict, cache=True, bigger_than_image=False, black_strain=False,
                 set_black_channel=None):
        '''If cache is passed, a ton of memory is used but things will go MUCH faster.'''
        self.image_name = image_name
        self.path_dict = path_dict
        self.cache= cache

        self.bioformats_xml = Bioformats_XML(self.path_dict['tif_folder'] + self.image_name)

        self.bigger_than_image = bigger_than_image

        # Add path to coalescence & annihilations
        image_name_without_extension = self.image_name.split('.')[0]
        self.annihilation_txt_path = self.path_dict['annihilation_folder'] + image_name_without_extension + '_annih.txt'
        self.coalescence_txt_path = self.path_dict['annihilation_folder']  + image_name_without_extension + '_coal.txt'

        self._brightfield_mask = None
        self._homeland_mask = None
        self._edges_mask = None
        self._doctored_edges_mask = None
        self._fluorescent_mask = None
        self._image = None

        # Domain sweeps
        self._labeled_domains = None
        self._unique_labeled_domains = None

        # Other useful stuff for data analysis
        self._image_coordinate_df = None
        self._fractions = None
        self._frac_df_list = None
        self._center_df = None

        # Information about the maximum radius of the data we care about
        self.homeland_edge_radius = None
        self.homeland_edge_radius_scaled = None

        self.max_radius = None
        self.max_radius_scaled = None

        self.black_strain = black_strain
        self.set_black_channel = set_black_channel

    def finish_setup(self):
        # Initialize rest of required stuff

        self.homeland_edge_radius = self.get_homeland_radius()
        if self.homeland_edge_radius is None:
            self.homeland_edge_radius_scaled = None
        else:
            self.homeland_edge_radius_scaled = self.homeland_edge_radius * self.get_scaling()

        self.max_radius = self.get_max_radius()
        self.max_radius_scaled = self.max_radius * self.get_scaling()

    def unitialize(self):
        self._brightfield_mask = None
        self._homeland_mask = None
        self._edges_mask = None
        self._doctored_edges_mask = None
        self._fluorescent_mask = None
        self._image = None

        # Other useful stuff for data analysis
        self._image_coordinate_df = None
        self._fractions = None
        self._frac_df_list = None
        self._center_df = None

        # Information about the maximum radius of the data we care about
        self.homeland_edge_radius = None
        self.homeland_edge_radius_scaled = None

        self.max_radius = None
        self.max_radius_scaled = None

    ###### Labeled Domains ####
    @property
    def labeled_domains(self):
        '''Each domain has the same label.'''
        if self._labeled_domains is None:
            try:
                temp_domains = ski.io.imread(self.path_dict['labeled_domains'] + self.image_name, plugin='tifffile')
                # Transform the colored domains into greyscale
                temp_domains_color = np.rollaxis(temp_domains, 0, 3)
                grey_domains = ski.color.rgb2gray(temp_domains_color)

                # Transform the greyscale labels into unique binary labels
                unique_greys = np.unique(grey_domains)
                final_labeled_domains = np.zeros(grey_domains.shape, dtype=np.int)
                count = 1
                for u in unique_greys:
                    if u != 0:
                        final_labeled_domains[grey_domains == u] = count
                        count += 1
            except IOError:
                print 'No labeled domains found...'
                return None

            if self.cache:
                self._labeled_domains = final_labeled_domains
                return self._labeled_domains
        else:
            return self._labeled_domains

    @labeled_domains.setter
    def labeled_domains(self, value):
        self._labeled_domains = value

    @labeled_domains.deleter
    def labeled_domains(self):
        del self._labeled_domains

    def get_edge_df(self, radius_start=0, radius_end=11, num_bins=300):
        labeled_domains = self.labeled_domains
        unique_labels = ski.measure.label(labeled_domains, neighbors=8, background=0) + 1 # Labels should go from 1 to infinity.

        cur_im_df = self.image_coordinate_df
        cur_im_df['domain_label'] = labeled_domains.ravel()
        cur_im_df['unique_label'] = unique_labels.ravel()

        # Only focus on the domains.
        nonzero_im_df = cur_im_df.loc[cur_im_df['domain_label'] != 0, :]

        # Get bins to average over each radius
        radius_bins = np.linspace(radius_start, radius_end, num=num_bins)
        mid_radius_bins = (radius_bins[:-1] + radius_bins[1:])/2.

        bin_cut = pd.cut(nonzero_im_df.radius_scaled, radius_bins, labels=mid_radius_bins)
        # Filter boundaries to a single point at each radius
        filtered_boundaries = nonzero_im_df.groupby(['unique_label', bin_cut]).agg(np.mean)
        filtered_boundaries.rename(columns={'radius_scaled':'radius_scaled_mean'}, inplace=True)

        filtered_boundaries.drop(['c', 'r', 'delta_r', 'delta_c', 'theta', 'radius'], axis=1, inplace=True)

        # Assign the initial radius, drop NaN's
        filtered_boundaries.dropna(inplace=True)
        filtered_boundaries.reset_index(inplace=True)

        # Make the radius_scaled a number...not a category. Or bad things happen.
        filtered_boundaries['radius_scaled'] = filtered_boundaries['radius_scaled'].astype(np.double)

        return filtered_boundaries, mid_radius_bins

    def get_domain_dfs(self, radius_start=0, radius_end=11, num_bins=300, theta_death=0.1):
        labeled_domains = self.labeled_domains
        unique_labels = ski.measure.label(labeled_domains, neighbors=8, background=0) + 1 # Labels should go from 1 to infinity.

        cur_im_df = self.image_coordinate_df
        cur_im_df['domain_label'] = labeled_domains.ravel()
        cur_im_df['unique_label'] = unique_labels.ravel()

        # Only focus on the domains.
        nonzero_im_df = cur_im_df.loc[cur_im_df['domain_label'] != 0, :]

        # Get bins to average over each radius
        radius_bins = np.linspace(radius_start, radius_end, num=num_bins)
        mid_radius_bins = (radius_bins[:-1] + radius_bins[1:])/2.

        bin_cut = pd.cut(nonzero_im_df.radius_scaled, radius_bins, labels=mid_radius_bins)
        # Filter boundaries to a single point at each radius
        filtered_boundaries = nonzero_im_df.groupby(['unique_label', bin_cut]).agg(np.mean)
        filtered_boundaries.rename(columns={'radius_scaled':'radius_scaled_mean'}, inplace=True)

        # Loop over domains, extract desired info
        domain_df = filtered_boundaries.reset_index().set_index(['domain_label'])
        delta_df_list = []

        for cur_domain, cur_domain_data in domain_df.groupby(level=0):
            # Group the domain data by the unique label of each edge.
            gb = cur_domain_data.groupby('unique_label')
            # There should be two indices in the group. Any more and something terrible has happened...
            df_list = []
            for n, d in gb:
                df_list.append(d.set_index('radius_scaled'))

            assert len(df_list) == 2, 'There should only be two edges per domain...I am getting ' + str(len(df_list))

            # Get deltaTheta at each radius from the distance between the two domains.
            delta_df = df_list[0] - df_list[1]
            delta_df['deltaX'] = np.sqrt(delta_df['r']**2 + delta_df['c']**2)
            delta_df['deltaX_scaled'] = self.get_scaling() * delta_df['deltaX']

            delta_df.reset_index(inplace=True)
            # Use trig to derive how the distance between the two points relates to deltaTheta.
            # Required as if theta changes from 2pi -> 0, bad things can happen when averaging.
            delta_df['delta_theta'] = 2 * np.arcsin(delta_df['deltaX_scaled']/(2*delta_df['radius_scaled']))

            delta_df.drop(['unique_label', 'c', 'r', 'delta_r', 'delta_c', 'radius', 'radius_scaled_mean', 'theta'],
                          axis=1, inplace=True)

            delta_df.dropna(inplace=True)

            # Get the maximum surviving radius
            initial_radius = np.min(delta_df['radius_scaled'])

            delta_df['initial_radius'] = initial_radius
            max_surviving_radius = np.max(delta_df['radius_scaled'])
            delta_df['max_radius'] = max_surviving_radius
            # Get deltaTheta at the max surviving radius
            final_deltaTheta = delta_df.iloc[-1]['delta_theta']


            # Get the initial angle
            initial_theta = delta_df['delta_theta'].iloc[0]
            delta_df['theta_o'] = initial_theta
            delta_df['theta_minus_theta_o'] = delta_df['delta_theta'] - delta_df['theta_o']

            # Based on the desired bins, if the domain went extinct, return
            # zero delta_theta. So, we probably have to reindex...

            delta_df.set_index('radius_scaled', inplace=True)
            delta_df = delta_df.reindex(index=mid_radius_bins)
            delta_df.reset_index(inplace=True)

            # If the domain goest extinct, set variables appropriately. Extinction is defined by a cutoff theta at the end.
            went_extinct = False
            if final_deltaTheta < theta_death:
               went_extinct = True
               delta_df.loc[delta_df['radius_scaled'] >= max_surviving_radius, ['delta_theta', 'deltaX', 'deltaX_scaled']] = 0
               delta_df.loc[delta_df['radius_scaled'] >= max_surviving_radius, ['theta_minus_theta_o']] = -initial_theta

            delta_df['theta_o'] = initial_theta
            delta_df['initial_radius'] = initial_radius
            delta_df['max_radius'] = max_surviving_radius
            delta_df['domain_label'] = cur_domain

            # Now calculate the difference in theta...I chose poor variable names

            delta_df['log_R_div_Ro'] = np.log(delta_df['radius_scaled']/delta_df['initial_radius'])

            delta_df_list.append(delta_df)

        combined_domains = pd.concat(delta_df_list)
        return combined_domains

    ###### Circular Mask ######
    @property
    def brightfield_mask(self):
        '''Returns the circle mask of brightfield. Takes a long time to run, so cache if possible.'''
        if self._brightfield_mask is None:
            try:
                temp_mask = ski.io.imread(self.path_dict['circle_folder'] + self.image_name, plugin='tifffile') > 0
            except IOError:
                print 'No circle mask found!'
                return None
            if self.cache:
                self._brightfield_mask = temp_mask
                return self._brightfield_mask
            else:
                return temp_mask
        else:
            return self._brightfield_mask

    @brightfield_mask.setter
    def brightfield_mask(self, value):
        self._brightfield_mask = value

    @brightfield_mask.deleter
    def brightfield_mask(self):
        del self._brightfield_mask

    ###### Homeland Mask ######
    @property
    def homeland_mask(self):
        '''Returns the circle mask of brightfield. Takes a long time to run, so cache if possible.'''
        if self._homeland_mask is None:
            try:
                temp_mask = ski.io.imread(self.path_dict['homeland_folder'] + self.image_name, plugin='tifffile') > 0
            except IOError:
                print 'No homeland mask found!'
                return None
            if self.cache:
                self._homeland_mask = temp_mask
                return self._homeland_mask
            else:
                return temp_mask
        else:
            return self._homeland_mask

    @homeland_mask.setter
    def homeland_mask(self, value):
        self._homeland_mask = value

    @homeland_mask.deleter
    def homeland_mask(self):
        del self._homeland_mask

    ###### center_df ######
    @property
    def center_df(self):
        '''Returns a dataframe with the center of the range expansion in image coordinates'''
        if self._center_df is None:
            temp_center = self.get_center()
            if self.cache:
                self._center_df = temp_center
                return self._center_df
            else:
                return temp_center
        else:
            return self._center_df

    @center_df.setter
    def center_df(self, value):
        self._center_df = value

    @center_df.deleter
    def center_df(self):
        del self._center_df

    ###### Edges Mask ######
    @property
    def edges_mask(self):
        '''Returns the edge binary image.'''
        if self._edges_mask is None:
            try:
                temp_mask = ski.io.imread(self.path_dict['edges_folder'] + self.image_name, plugin='tifffile') > 0
            except IOError:
                print 'No edges mask found!'
                return None
            if self.cache:
                self._edges_mask = temp_mask
                return self._edges_mask
            else:
                return temp_mask
        else:
            return self._edges_mask

    @edges_mask.setter
    def edges_mask(self, value):
        self._edges_mask = value

    @edges_mask.deleter
    def edges_mask(self):
        del self._edges_mask

    ######## Doctored Edges Mask ########
    @property
    def doctored_edges_mask(self):
        '''Returns the edge binary image.'''
        if self._doctored_edges_mask is None:
            try:
                temp_mask = ski.io.imread(self.path_dict['doctored_edges_folder'] + self.image_name, plugin='tifffile') > 0
            except IOError:
                print 'No doctored edges mask found!'
                return None
            if self.cache:
                self._doctored_edges_mask = temp_mask
                return self._doctored_edges_mask
            else:
                return temp_mask
        else:
            return self._doctored_edges_mask

    @doctored_edges_mask.setter
    def doctored_edges_mask(self, value):
        self._doctored_edges_mask = value

    @doctored_edges_mask.deleter
    def doctored_edges_mask(self):
        del self._doctored_edges_mask

    ######### Fluorescent Masks ########
    @property
    def fluorescent_mask(self):
        '''Returns the mask of each channel.'''
        if self._fluorescent_mask is None:
            try:
                temp_mask = ski.io.imread(self.path_dict['masks_folder'] + self.image_name, plugin='tifffile') > 0
            except IOError:
                print 'No channel masks found!'
                return None
            if self.set_black_channel is not None:
                indices = np.ones(temp_mask.shape[0]) > 0
                indices[self.set_black_channel] = False
                temp_mask = temp_mask[indices, :, :]
            if self.black_strain or self.set_black_channel is not None:
                # Create a black color...the absence of the other two
                black_channel = ~np.any(temp_mask, axis=0)

                insert_location = None
                if self.set_black_channel is not None:
                    insert_location = self.set_black_channel
                else:
                    insert_location = temp_mask.shape[0]

                temp_mask = np.insert(temp_mask, insert_location, black_channel, axis=0)

            if self.cache:
                self._fluorescent_mask = temp_mask
                return self._fluorescent_mask
            else:
                return temp_mask
        else:
            return self._fluorescent_mask

    @fluorescent_mask.setter
    def fluorescent_mask(self, value):
        self.fluorescent_mask = value

    @fluorescent_mask.deleter
    def fluorescent_mask(self):
        del self.fluorescent_mask

    ######## Image ######
    @property
    def image(self):
        '''Returns the original image.'''
        if self._image is None:
            try:
                temp_image= ski.io.imread(self.path_dict['tif_folder'] + self.image_name, plugin='tifffile')
            except IOError:
                print 'No original image found! This is weird...'
                return None
            if self.cache:
                self._image= temp_image
                return self._image
            else:
                return temp_image
        else:
            return self._image

    @image.setter
    def image(self, value):
        self._image = value

    @image.deleter
    def image(self):
        del self._image

    ###### Fractions: necessary to deal with overlap sadly ######
    @property
    def fractions(self):
        '''Returns the color fractions.'''
        if self._fractions is None:
            temp_fractions = self.get_fractions_mask()
            if self.cache:
                self._fractions = temp_fractions
            return temp_fractions
        else:
            return self._fractions

    @fractions.setter
    def fractions(self, value):
        self._fractions = value

    @fractions.deleter
    def fractions(self):
        del self._fractions

    @property
    def frac_df(self):
        if self._frac_df_list is None:
            temp_list = self.get_frac_df()
            if self.cache:
                self._frac_df_list = temp_list
            return temp_list
        else:
            return self._frac_df_list

    @frac_df.setter
    def frac_df(self, value):
        self._frac_df_list = value

    @frac_df.deleter
    def frac_df(self):
        del self._frac_df_list

    ####### Image Coordinate df ####
    @property
    def image_coordinate_df(self):
        if self._image_coordinate_df is None:
            temp_coordinate_df = self.get_image_coordinate_df()
            if self.cache:
                self.image_coordinate_df = temp_coordinate_df
            return temp_coordinate_df
        else:
            return self._image_coordinate_df

    @image_coordinate_df.setter
    def image_coordinate_df(self, value):
        self._image_coordinate_df = value

    @image_coordinate_df.deleter
    def image_coordinate_df(self):
        del self._image_coordinate_df

    def get_image_coordinate_df(self):
        """Returns image coordinates in r and theta. Uses the center of the brightfield mask
        as the origin."""
        rows = np.arange(0, self.image.shape[1])
        columns = np.arange(0, self.image.shape[2])
        rmesh, cmesh = np.meshgrid(rows, columns, indexing='ij')

        r_ravel = rmesh.ravel()
        c_ravel = cmesh.ravel()

        df = pd.DataFrame(data={'r': r_ravel, 'c': c_ravel})

        cur_center_df = self.center_df
        df['delta_r'] = df['r'] - cur_center_df['r', 'mean'].values
        df['delta_c'] = df['c'] - cur_center_df['c', 'mean'].values
        df['radius'] = (df['delta_r']**2. + df['delta_c']**2.)**0.5
        df['radius_scaled'] = df['radius'] * self.get_scaling()
        df['theta'] = np.arctan2(df['delta_r'], df['delta_c'])

        return df

    def bin_image_coordinate_r_df(self, df):
        max_r_ceil = np.floor(df['radius'].max())
        bins = np.arange(0, max_r_ceil+ 2 , 1.5)
        groups = df.groupby(pd.cut(df.radius, bins))
        mean_groups = groups.agg('mean')
        # Assign the binning midpoints...
        mean_groups['radius_midbin'] = (bins[1:] + bins[:-1])/2.
        mean_groups['radius_midbin_scaled'] = mean_groups['radius_midbin'] * self.get_scaling()

        return mean_groups, bins

    def bin_theta_at_r_df(self, df, r, delta_x=1.5):
        """Assumes that the df has image_coordinate structure."""
        theta_df_list = []

        delta_theta = delta_x / float(r)
        theta_bins = np.arange(-np.pi, np.pi + delta_theta, delta_theta)
        theta_bins = theta_bins[:-1]

        # First get the theta at the desired r; r should be an int
        theta_df = df.query('(radius >= @r - 0.75) & (radius < @r + 0.75)')
        # TODO Below is a very slow line, try to speed it up

        theta_cut = pd.cut(theta_df['theta'], theta_bins)
        groups = theta_df.groupby(theta_cut)
        mean_df = groups.agg('mean')
        # Check for nan's
        if not mean_df[mean_df.isnull().any(axis=1)].empty > 0:
            print 'theta binning in bin_theta_at_r_df is producing NaN at r=' +str(r) + ', delta_x=' + str(delta_x) + \
                  ' name= ' + self.image_name
            print mean_df[mean_df.isnull().any(axis=1)]

        return mean_df, theta_bins


    ####### Main Functions #######

    def get_average_fractions_black_corrected(self, r_steps):
        # Get the fractions in steps of 1.5pixels

        cur_masks = self.fluorescent_mask
        num_channels = cur_masks.shape[0]

        mask_df =  self.image_coordinate_df
        count = 0
        for mask in cur_masks:
            string = 'ch' + str(count)
            mask_df[string] = mask.ravel()
            count += 1

        frac_list = []

        for cur_r in r_steps:
            # It would make more sense to go from the fluorescence directly, but this is a shortcut
            masks_binned, theta = self.bin_theta_at_r_df(mask_df, cur_r)
            # Get average overlap

            domains = masks_binned.loc[:, 'ch0':'ch' + str(num_channels - 1)].values
            domains = domains > 0
            overlap_bool = np.zeros(domains.shape[0])
            for i in range(num_channels):
                for j in range(i + 1, num_channels):
                    overlap_bool = np.logical_or(overlap_bool, domains[:, i] & domains[:, j])
            starts, stops, lengths = contiguous_regions(overlap_bool)

            average_percolor_overlap = np.rint(np.mean(lengths)/2)


            # Now adjust the black region appropriately...
            black_index = None
            if self.set_black_channel:
                black_index = self.set_black_channel
            else: # Assume the black domain is the third one here.
                black_index = num_channels - 1
            black_domain = domains[:, black_index]


            starts, stops, lengths = contiguous_regions(black_domain)

            if lengths.shape[0] == 0:
                black_domain[:] = True
            else:

                starts -= average_percolor_overlap
                stops += average_percolor_overlap

                # Now recreate the array
                for cur_start, cur_stop in zip(starts, stops):

                    if cur_start < 0:
                        black_domain[cur_start:] = True
                        cur_start = 0

                    num_elements = black_domain.shape[0]

                    if cur_stop > num_elements:
                        black_domain[0:cur_stop % num_elements] = True
                        cur_stop = num_elements

                    print cur_start, cur_stop

                    black_domain[cur_start:cur_stop] = True

                # Now reinsert it
                domains[:, black_index] = black_domain

            # Now calculate the fractions
            totals = domains.sum(axis=1, dtype=np.float64)
            fractions = domains/totals[:, None]

            fractions = fractions.mean(axis=0)

            frac_list.append(fractions)

        return np.array(frac_list)

    def get_domain_sizes_scaled_at_radius(self, pixel_radius):
        cur_masks = self.fluorescent_mask
        num_channels = cur_masks.shape[0]

        mask_df =  self.image_coordinate_df
        count = 0
        for mask in cur_masks:
            string = 'ch' + str(count)
            mask_df[string] = mask.ravel()
            count += 1

        masks_binned, theta = self.bin_theta_at_r_df(mask_df, pixel_radius)

        # Determine the length of the domains in each channel...
        domains = masks_binned.loc[:, 'ch0':'ch' + str(num_channels - 1)].values
        domains = domains > 0

        domain_dict = {}

        delta_theta = theta[1] - theta[0]

        domain_df_list = []

        for ch_num in range(domains.shape[1]):
            cur_domain = domains[:, ch_num]
            starts, stops, lengths = contiguous_regions(cur_domain)

            physical_radius = pixel_radius * self.get_scaling()

            lengths_scaled = physical_radius * delta_theta * lengths

            domain_dict['channel'] = ch_num
            domain_dict['lengths_scaled'] = lengths_scaled
            domain_dict['angular_width'] = 2*np.pi*(lengths / float(cur_domain.shape[0]))
            domain_dict['pixel_radius'] = pixel_radius

            domain_df = pd.DataFrame(domain_dict)

            domain_df_list.append(domain_df)

        combined_domain_df = pd.concat(domain_df_list)
        return combined_domain_df

    def get_biorep_name(self):
        """Assumes that in the name, bioSTUFF, STUFF is the replicate name."""
        name = self.image_name
        after_bio = name.split('bio')
        bio_name = None
        if len(after_bio) == 2:
            bio_name = after_bio[1].split('_')[0]
        else:
            bio_name = 'no_bio_rep_info'

        return bio_name.lower()

    #### Dealing with fractions: core of this analysis ####

    def get_fractions_mask(self):
        cur_channel_mask = self.fluorescent_mask
        if cur_channel_mask is not None:
            sum_mask = np.zeros((cur_channel_mask.shape[1], cur_channel_mask.shape[2]))
            for i in range(cur_channel_mask.shape[0]):
                sum_mask += cur_channel_mask[i, :, :]

            # Now divide each channel by the sum
            fractions = cur_channel_mask / sum_mask.astype(np.float)
            fractions[np.isnan(fractions)] = 0
            return fractions
        else:
            print 'Cannot determine color fractions because there is no channel mask.'
            return None

    def get_frac_df(self):

        cur_fractions = self.fractions
        cur_im_coordinate =  self.image_coordinate_df
        count = 0
        for frac in cur_fractions:
            string = 'ch' + str(count)
            cur_im_coordinate[string] = frac.ravel()
            count += 1

        return cur_im_coordinate


    def get_fracs_at_radius(self):
        """Gets fractions binned at all radii."""
        cur_frac_df = self.frac_df
        binned_by_radius, bins = self.bin_image_coordinate_r_df(cur_frac_df)

        return binned_by_radius

    ##### Other stuff #####

    @staticmethod
    def sem(x):
        return sp.stats.sem(x, ddof=2)

    def get_center(self):
        """Returns the mean center and the standard error of the mean"""
        cur_homeland_mask = self.homeland_mask
        if cur_homeland_mask is None:
            return None

        center_list = []
        for i in range(cur_homeland_mask.shape[0]):
            cur_image = cur_homeland_mask[i, :, :]
            label_image = ski.measure.label(cur_image, neighbors=8)
            props = ski.measure.regionprops(label_image)
            for p in props:
                # There should only be one property
                center_list.append(p['centroid'])
        center_list = np.asarray(center_list)
        center_df = pd.DataFrame(data = center_list, columns=('r', 'c'))
        result_df = center_df.groupby(lambda x: 0).agg([np.mean, Image_Set.sem])

        return result_df

    def get_max_radius(self):
        cur_brightfield_mask = self.brightfield_mask
        diameter_list = []
        for i in range(cur_brightfield_mask.shape[0]):
            cur_image = cur_brightfield_mask[i, :, :]
            # Find maximum diameter
            r, c = np.where(cur_image)
            diameter_r = np.float(r.max() - r.min())
            diameter_c = np.float(c.max() - c.min())
            # We need both of these checks in case the expansion is bigger than the image in one dimension...
            desired_diameter = max(diameter_r, diameter_c)
            diameter_list.append(desired_diameter)
        diameter_list = np.array(diameter_list)
        # Now find the mean radius
        max_radius = int(np.floor(diameter_list.mean()/2))
        return max_radius

    def get_homeland_radius(self):
        cur_homeland_mask = self.homeland_mask
        if cur_homeland_mask is not None:
            diameter_list = []
            for i in range(cur_homeland_mask.shape[0]):
                cur_image = cur_homeland_mask[i, :, :]
                # Find maximum diameter
                r, c = np.where(cur_image)
                diameter = np.float(r.max() - r.min())
                diameter_list.append(diameter)
            diameter_list = np.array(diameter_list)
            # Now find the mean radius
            homeland_radius = int(np.floor(diameter_list.mean()/2))
            return homeland_radius
        else:
            return None


    def get_local_hetero_df(self):

        cur_frac_df = self.frac_df

        num_channels = self.fluorescent_mask.shape[0]
        start_string = 'ch0'
        finish_string = 'ch' + str(num_channels -1)

        channel_data = cur_frac_df.loc[:, start_string:finish_string].values

        local_h = np.sum(channel_data * (1 - channel_data), axis=1)
        cur_frac_df['h'] = local_h

        return cur_frac_df

    def get_nonlocal_quantity(self, quantity, r, delta_x=1.5, i=None, j=None, calculate_overlap=False):
        """Calculates nonlocal information based on the quantity keyword."""
        cur_frac_df = self.frac_df

        theta_df_list, theta_bins = self.bin_theta_at_r_df(cur_frac_df, r, delta_x = delta_x)
        theta_step = theta_bins[1] - theta_bins[0]
        num_channels = self.fluorescent_mask.shape[0]

        ### Variables required for most functions ###
        values = None
        convolve_list = None

        ### Variables required for Fij_sym ###
        convolve_list_1 = None
        convolve_list_2 = None
        values_1 = None
        values_2 = None

        ### Grab desired data ###

        if quantity == 'hetero':
            start_string = 'ch0'
            finish_string = 'ch' + str(num_channels -1)

            values = theta_df_list.loc[:, start_string:finish_string].values
            convolve_list = values.copy()
        elif quantity== 'Ftot':
            start_string = 'ch0'
            finish_string = 'ch' + str(num_channels -1)

            values = theta_df_list.loc[:, start_string:finish_string].values
            convolve_list = values.copy()
        elif quantity == 'Fij':
            i_string = 'ch' + str(i)
            j_string = 'ch' + str(j)

            values = theta_df_list.loc[:, i_string].values
            convolve_list = theta_df_list.loc[:, j_string].values
        elif quantity == 'Fij_sym':
            i_string = 'ch' + str(i)
            j_string = 'ch' + str(j)

            values_1 = theta_df_list.loc[:, i_string].values
            convolve_list_1 = theta_df_list.loc[:, j_string].values

            values_2 = theta_df_list.loc[:, j_string].values
            convolve_list_2 = theta_df_list.loc[:, i_string].values

        # Number of points to calcualte
        num_points = theta_df_list.values.shape[0]
        delta_theta_list = -1.*np.ones(num_points, dtype=np.double)
        mean_list = -1.*np.ones(num_points, dtype=np.double)
        for i in range(num_points):
            if i == 0:
                delta_theta_list[i] = 0
            else:
                delta_theta_list[i] = delta_theta_list[i - 1] + theta_step

            if quantity =='hetero':
                # Calculate the heterozygosity
                multiplied = values * (1 - convolve_list)
                av_channel_hetero = multiplied.mean(axis=0)
                h = av_channel_hetero.sum()
                mean_list[i] = h
                convolve_list = np.roll(convolve_list, 1, axis=0)

            elif quantity =='Ftot':
                multiplied = values * convolve_list
                av_channel_hetero = multiplied.mean(axis=0)
                Ftot = av_channel_hetero.sum()
                mean_list[i] = Ftot
                convolve_list = np.roll(convolve_list, 1, axis=0)

            elif quantity == 'Fij':
                multiplied = values*convolve_list
                av_Fij = multiplied.mean(axis=0)
                mean_list[i] = av_Fij
                convolve_list = np.roll(convolve_list, 1, axis=0)

            elif quantity == 'Fij_sym':
                multiplied = values_1 * convolve_list_1 + values_2 * convolve_list_2
                av_Fij_sym = multiplied.mean(axis=0)
                mean_list[i] = av_Fij_sym

                convolve_list_1 = np.roll(convolve_list_1, 1, axis=0)
                convolve_list_2 = np.roll(convolve_list_2, 1, axis=0)

        # Return theta between -pi and pi
        delta_theta_list[delta_theta_list > np.pi] -= 2*np.pi

        # Now sort based on theta
        sorted_indices = np.argsort(delta_theta_list)
        delta_theta_list = delta_theta_list[sorted_indices]
        mean_list = mean_list[sorted_indices]

        dict_to_return = {}
        dict_to_return['result'] = mean_list
        dict_to_return['theta_list'] = delta_theta_list
        dict_to_return['average_overlap'] = None

        # Calculate overlap, if necessary
        if calculate_overlap:
            # Get the overlap df at that radius
            overlap_df = self.get_overlap_df(2)
            #TODO: delta_x should probably be defined as a constant in this package
            delta_x = 1.5
            overlap_at_radius = overlap_df.query('(radius >= @r - @delta_x/2.) & (radius < @r + @delta_x/2.)')
            fraction_overlap = float(np.sum(overlap_at_radius['overlap']))/overlap_at_radius.shape[0]
            # Get the fraction in radians
            average_angular_overlap = fraction_overlap * 2*np.pi
            dict_to_return['average_overlap'] = average_angular_overlap
        return dict_to_return

    def get_nonlocal_hetero(self, r, delta_x = 1.5, **kwargs):
        """Calculates the heterozygosity at every theta."""
        return self.get_nonlocal_quantity('hetero', r, delta_x = delta_x, **kwargs)

    def get_nonlocal_Ftot(self, r, delta_x = 1.5, **kwargs):
        """Calculates Ftot every theta."""
        return self.get_nonlocal_quantity('Ftot', r, delta_x = delta_x, **kwargs)

    def get_nonlocal_Fij(self, r, i=None, j=None, delta_x = 1.5, **kwargs):
        """Calculates F_ij at every theta along with its error."""
        return self.get_nonlocal_quantity('Fij', r, delta_x=delta_x, i=i, j=j, **kwargs)

    def get_nonlocal_Fij_sym(self, r, i=None, j=None, delta_x = 1.5, **kwargs):
        """Calculates F_ij at every theta along with its error."""
        return self.get_nonlocal_quantity('Fij_sym', r, delta_x=delta_x, i=i, j=j, **kwargs)

    def get_local_hetero_mask(self):
        local_hetero_mask = np.zeros((self.fractions.shape[1], self.fractions.shape[2]))

        for i in range(self.fractions.shape[0]):
            for j in range(self.fractions.shape[0]):
                if i != j:
                    draw_different = self.fractions[i] * self.fractions[j]
                    local_hetero_mask += draw_different

        return local_hetero_mask

    #### Overlap Images ####

    def get_overlap_image(self, num_overlap):
        """Return a mask stating if there is an overlap of more than num_overlap colors."""
        cur_channel_mask = self.fluorescent_mask
        sum_mask = np.zeros((cur_channel_mask.shape[1], cur_channel_mask.shape[2]))
        for i in range(cur_channel_mask.shape[0]):
            sum_mask += cur_channel_mask[i, :, :]
        edges = sum_mask >= num_overlap

        return edges


    def get_overlap_df(self, num_overlap):
        """Returns if there is an overlap or not."""
        edge_image = self.get_overlap_image(num_overlap)

        edge_df = self.image_coordinate_df.copy()
        edge_df['overlap'] = edge_image.ravel()
        edge_df = edge_df[edge_df['radius'] < self.max_radius]
        return edge_df

    def get_edge_skeleton(self):
        '''Gets the pruned skeleton for any edges.'''
        all_edges = self.get_overlap_image(2)
        # Filter out the homeland; just multiply by the inverse mask!
        all_edges = np.logical_and(all_edges, np.logical_not(self.homeland_mask))
        all_edges = ski.morphology.closing(all_edges, ski.morphology.square(5))

        skeleton = mh.thin(all_edges > 0)
        pruned = pruning(skeleton, 20)

        return pruned

    #### Annihilations and Coalescences ####
    def get_annih_and_coal(self):
        cur_image_coordinate_df = self.image_coordinate_df

        cur_annihilations = pd.read_csv(self.annihilation_txt_path, sep='\t')
        cur_coalescences = pd.read_csv(self.coalescence_txt_path, sep='\t')
        annihilations_df = pd.merge(cur_annihilations, cur_image_coordinate_df, on=['r', 'c'])
        coalescence_df = pd.merge(cur_coalescences, cur_image_coordinate_df, on=['r', 'c'])

        return annihilations_df, coalescence_df

    # Utility functions

    def remove_small(self, input_image, size_cutoff=20):
        '''Scikit image does not appear to be doing this correctly...I did it myself.'''

        output_image = np.zeros_like(input_image, dtype=np.bool)

        labeled_image = ski.measure.label(input_image)
        regionprops = ski.measure.regionprops(labeled_image)
        for p in regionprops:
            if p.area > size_cutoff:
                coords = p.coords
                output_image[coords[:, 0], coords[:, 1]] = True

        return output_image

    def get_scaling(self):
        '''Assumes x & y pixel scaling are the same'''
        scaling_str = self.bioformats_xml.pixel_nodes[0].attrib['PhysicalSizeX']
        return float(scaling_str)

    # Plotting functions
    def plot_local_hetero(self):
        '''Make a simple plot of the local heterozygosity. Useful for diagnostic purposes.'''
        hetero_df = self.get_local_hetero_df()
        binned_hetero, bins = self.bin_image_coordinate_r_df(hetero_df)
        # Add a scaled radius row
        binned_hetero['radius_midbin_scaled'] = binned_hetero['radius_midbin'] * self.get_scaling()
        plt.loglog(binned_hetero['radius_midbin_scaled'], binned_hetero['h', 'mean'])
        plt.xlabel('Radius (mm)')
        plt.ylabel(r'$H(r)$')


class Bioformats_XML(object):
    def __init__(self, path):
        self.path = path
        self.xml_str = None
        self.xml_root = None
        self.image_nodes = None
        self.pixel_nodes = None
        self.channel_nodes = None

        self.finish_setup()


    def finish_setup(self):
        self.set_bioformats_xml()
        self.setup_xml()

    def set_bioformats_xml(self):
        with ti.TiffFile(self.path) as tif:
            first_page = tif.pages[0]
            for tag in first_page.tags.values():
                    if tag.name == 'image_description':
                        self.xml_str = tag.value

    def setup_xml(self):
        self.xml_root = ET.fromstring(self.xml_str)
        self.image_nodes = []
        self.pixel_nodes = []
        self.channel_nodes = []
        for child in self.xml_root:
            if child.tag.endswith('Image'):
                self.image_nodes.append(child)
                for grandchild in child:
                    if grandchild.tag.endswith('Pixels'):
                        self.pixel_nodes.append(grandchild)
                        temp_channels = []
                        for greatchild in grandchild:
                            if greatchild.tag.endswith('Channel'):
                                temp_channels.append(greatchild)
                        self.channel_nodes.append(temp_channels)

##### Pruning Functions: Pulled from online. I should really learn this at some point! #####

def branchedPoints(skel):
    branch1=np.array([[2, 1, 2], [1, 1, 1], [2, 2, 2]])
    branch2=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch3=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]])
    branch4=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    branch5=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    branch6=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    branch7=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    branch8=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    branch9=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    br1=mh.morph.hitmiss(skel,branch1)
    br2=mh.morph.hitmiss(skel,branch2)
    br3=mh.morph.hitmiss(skel,branch3)
    br4=mh.morph.hitmiss(skel,branch4)
    br5=mh.morph.hitmiss(skel,branch5)
    br6=mh.morph.hitmiss(skel,branch6)
    br7=mh.morph.hitmiss(skel,branch7)
    br8=mh.morph.hitmiss(skel,branch8)
    br9=mh.morph.hitmiss(skel,branch9)
    return br1+br2+br3+br4+br5+br6+br7+br8+br9

def endPoints(skel):
    endpoint1=np.array([[0, 0, 0],
                        [0, 1, 0],
                        [2, 1, 2]])

    endpoint2=np.array([[0, 0, 0],
                        [0, 1, 2],
                        [0, 2, 1]])

    endpoint3=np.array([[0, 0, 2],
                        [0, 1, 1],
                        [0, 0, 2]])

    endpoint4=np.array([[0, 2, 1],
                        [0, 1, 2],
                        [0, 0, 0]])

    endpoint5=np.array([[2, 1, 2],
                        [0, 1, 0],
                        [0, 0, 0]])

    endpoint6=np.array([[1, 2, 0],
                        [2, 1, 0],
                        [0, 0, 0]])

    endpoint7=np.array([[2, 0, 0],
                        [1, 1, 0],
                        [2, 0, 0]])

    endpoint8=np.array([[0, 0, 0],
                        [2, 1, 0],
                        [1, 2, 0]])

    ep1=mh.morph.hitmiss(skel,endpoint1)
    ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3)
    ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5)
    ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7)
    ep8=mh.morph.hitmiss(skel,endpoint8)
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    return ep

def pruning(skeleton, size):
    '''remove iteratively end points "size"
       times from the skeleton
    '''
    for i in range(0, size):
        endpoints = endPoints(skeleton)
        endpoints = np.logical_not(endpoints)
        skeleton = np.logical_and(skeleton,endpoints)
    return skeleton
