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

class Multi_Experiment(object):
    """Assumes that you don't have enough memory to store everything. Writes things to disk
       as we go."""

    def __init__(self, experiment_list, complete_im_set_list):
        # Assumes that the images in the experiments are setup appropriately.
        self.experiment_list = experiment_list
        self.complete_im_sets_list = complete_im_set_list
        self.hetero_r_list = [2.5, 3, 4, 6, 8, 10] # Radii used to compare heterozygosity
        self.num_theta_bins_list = [500, 700, 1000, 1000, 1500, 1500] # Bins at each radius; larger radii allow more bins

    def write_hetero_to_disk(self):
        h_list = []
        for experiment, complete_im_sets in zip(self.experiment_list, self.complete_im_sets_list):

            h_info = {}
            h_info['r_list'] = self.hetero_r_list
            h_info['num_theta_bins_list'] = self.num_theta_bins_list

            # Make new directory for this experiment...give experiment a name
            for r, theta_bins in zip(self.hetero_r_list, self.num_theta_bins_list):
                h = experiment.get_nonlocal_hetero_averaged(complete_im_sets, r, num_theta_bins=theta_bins,
                                                            skip_grouping=True)
                h_list.append(h)
            h_info['h_list'] = h_list
            with open(experiment.title + '_hetero.pkl', 'wb') as fi:
                pkl.dump(h_info, fi)

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

        self.image_set_list = None

        self.finish_setup(**kwargs)

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

            frac_list.append(fracs)

        frac_concat = pd.concat(frac_list)
        return frac_concat


    @staticmethod
    def bin_annih_or_coal(df, bins):
        """Calculates cumulative sums for annihilations/coalescences along bins which can later be averaged."""
        df['count'] = 1
        cut = pd.cut(df['radius_scaled'], bins)
        gb = df.groupby(cut)
        mean_bins = gb.agg('sum')
        mean_bins.loc[pd.isnull(mean_bins['count']), 'count'] = 0
        mean_bins['cumsum'] = np.cumsum(mean_bins['count'])

        # Add radius_midbin to each...
        mean_bins['radius_scaled_midbin'] = (bins[:-1] + bins[1:])/2.

        return mean_bins

    def get_cumulative_average_annih_coal(self, im_set_indices_to_use, min_radius_scaled = 2.5, max_radius_scaled = 11,
                                          num_bins = 500):
        new_r_bins = np.linspace(min_radius_scaled, max_radius_scaled, num_bins)
        binned_annih_list = []
        binned_coal_list = []
        for index in im_set_indices_to_use:
            cur_im_set = self.image_set_list[index]
            annih, coal = cur_im_set.get_annih_and_coal()

            binned_annih = Range_Expansion_Experiment.bin_annih_or_coal(annih, new_r_bins)
            binned_coal  = Range_Expansion_Experiment.bin_annih_or_coal(coal, new_r_bins)

            binned_annih['index'] = index
            binned_coal['index'] = index
            binned_annih['bio_replicate'] = cur_im_set.get_biorep_name()
            binned_coal['bio_replicate'] = cur_im_set.get_biorep_name()

            binned_annih_list.append(binned_annih)
            binned_coal_list.append(binned_coal)

        combined_annih = pd.concat(binned_annih_list, join='outer')
        combined_coal = pd.concat(binned_coal_list)

        return combined_annih, combined_coal

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

    def get_nonlocal_quantity_averaged(self, nonlocal_quantity, im_sets_to_use, r_scaled, num_theta_bins=250, delta_x=1.5,
                                       skip_grouping=False, **kwargs):
        df_list = []
        standard_theta_bins = np.linspace(-np.pi, np.pi, num_theta_bins)
        midbins = (standard_theta_bins[1:] + standard_theta_bins[0:-1])/2.

        for im_set_index in im_sets_to_use:
            cur_im_set = self.image_set_list[im_set_index]
            cur_scaling = cur_im_set.get_scaling()

            desired_r = np.around(r_scaled / cur_scaling)
            result, theta_list = None, None
            cur_df = None
            if nonlocal_quantity == 'hetero':
                result, theta_list = cur_im_set.get_nonlocal_hetero(desired_r, delta_x = delta_x, **kwargs)
                cur_df = pd.DataFrame(data={'h':result, 'theta': theta_list})
            elif nonlocal_quantity == 'Ftot':
                result, theta_list = cur_im_set.get_nonlocal_Ftot(desired_r, delta_x = delta_x, **kwargs)
                cur_df = pd.DataFrame(data={'Ftot':result, 'theta': theta_list})
            elif nonlocal_quantity == 'Fij':
                result, theta_list = cur_im_set.get_nonlocal_Fij(desired_r, delta_x = delta_x, **kwargs)
                cur_df = pd.DataFrame(data={'Fij':result, 'theta': theta_list})
            elif nonlocal_quantity == 'Fij_sym':
                result, theta_list = cur_im_set.get_nonlocal_Fij_sym(desired_r, delta_x = delta_x, **kwargs)
                cur_df = pd.DataFrame(data={'Fij_sym':result, 'theta': theta_list})

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
    def __init__(self, image_name, path_dict, cache=True, bigger_than_image=True):
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

    def finish_setup(self):
        # Initialize rest of required stuff

        self.homeland_edge_radius = self.get_homeland_radius()
        self.homeland_edge_radius_scaled = self.homeland_edge_radius * self.get_scaling()

        self.max_radius = self.get_max_radius()
        self.max_radius_scaled = self.max_radius * self.get_scaling()

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

    ####### Main Functions #######

    def get_biorep_name(self):
        """Assumes that in the name, bioSTUFF, STUFF is the replicate name."""
        name = self.image_name
        after_bio = name.split('bio')
        bio_name = None
        if len(after_bio) == 2:
            bio_name = after_bio.split('_')[0]
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
        max_radius = None
        if self.bigger_than_image:
            # Find the minimum distance to the edge of the innoculation; this will be a straight line
            # from the center to an edge
            cur_center = self.center_df
            cur_homeland_mask = self.homeland_mask
            max_r = cur_homeland_mask.shape[1]
            max_c = cur_homeland_mask.shape[2]

            dist_to_top = max_r - cur_center['r', 'mean']
            dist_to_right = max_c - cur_center['c', 'mean']
            dist_to_bottom = cur_center['r', 'mean']
            dist_to_left = cur_center['c', 'mean']

            max_radius = min(dist_to_top.values, dist_to_right.values, dist_to_bottom.values, dist_to_left.values)
        else:
            cur_brightfield_mask = self.brightfield_mask

            diameter_list = []
            for i in range(cur_brightfield_mask.shape[0]):
                cur_image = cur_brightfield_mask[i, :, :]
                # Find maximum diameter
                r, c = np.where(cur_image)
                diameter = np.float(r.max() - r.min())
                diameter_list.append(diameter)
            diameter_list = np.array(diameter_list)
            # Now find the mean radius
            max_radius = int(np.floor(diameter_list.mean()/2))
        return max_radius

    def get_homeland_radius(self):
        cur_homeland_mask = self.homeland_mask
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
        theta_df = df.query('(radius >= @r - @delta_x/2.) & (radius < @r + @delta_x/2.)')
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

    def get_local_hetero_df(self):

        cur_frac_df = self.frac_df

        num_channels = self.fluorescent_mask.shape[0]
        start_string = 'ch0'
        finish_string = 'ch' + str(num_channels -1)

        channel_data = cur_frac_df.loc[:, start_string:finish_string].values

        local_h = np.sum(channel_data * (1 - channel_data), axis=1)
        cur_frac_df['h'] = local_h

        return cur_frac_df

    def get_nonlocal_quantity(self, quantity, r, delta_x=1.5, i=None, j=None):
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

        return mean_list, delta_theta_list


    def get_nonlocal_hetero(self, r, delta_x = 1.5):
        """Calculates the heterozygosity at every theta."""
        return self.get_nonlocal_quantity('hetero', r, delta_x = delta_x)

    def get_nonlocal_Ftot(self, r, delta_x = 1.5):
        """Calculates Ftot every theta."""
        return self.get_nonlocal_quantity('Ftot', r, delta_x = delta_x)

    def get_nonlocal_Fij(self, r, i=None, j=None, delta_x = 1.5):
        """Calculates F_ij at every theta along with its error."""
        return self.get_nonlocal_quantity('Fij', r, delta_x=delta_x, i=i, j=j)

    def get_nonlocal_Fij_sym(self, r, i=None, j=None, delta_x = 1.5):
        """Calculates F_ij at every theta along with its error."""
        return self.get_nonlocal_quantity('Fij_sym', r, delta_x=delta_x, i=i, j=j)

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
        #sum_mask counts how many different colors are at each pixel
        cur_channel_mask = self.fluorescent_mask
        sum_mask = np.zeros((cur_channel_mask.shape[1], cur_channel_mask.shape[2]))
        for i in range(cur_channel_mask.shape[0]):
            sum_mask += cur_channel_mask[i, :, :]
        edges = sum_mask >= num_overlap

        return edges


    def get_overlap_df(self, num_overlap):
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
