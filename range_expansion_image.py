__author__ = 'bryan'
import skimage as ski
import skimage.io
import skimage.measure
import matplotlib.pyplot as plt
import tifffile as ti
import xml.etree.ElementTree as ET
import glob
import os
import pandas as pd
import numpy as np
import scipy as sp

class Range_Expansion_Experiment():
    def __init__(self, base_folder):
        self.path_dict = {}
        self.path_dict['circle_folder'] = base_folder + 'circle_radius/'
        self.path_dict['edges_folder'] = base_folder + 'edges/'
        self.path_dict['doctored_edges_folder'] = base_folder + 'edges_doctored/'
        self.path_dict['masks_folder'] = base_folder + 'masks/'
        self.path_dict['tif_folder'] = base_folder + 'tif/'

        self.image_set_list = None

        self.finish_setup()

    def finish_setup(self):
        # Get a list of all images
        tif_paths = glob.glob(self.path_dict['tif_folder'] + '*.ome.tif')
        tif_paths.sort()

        # Now setup each set of images
        image_names = [os.path.basename(z) for z in tif_paths]

        self.image_set_list = []

        for cur_name in image_names:
            im_set = Image_Set(cur_name, self.path_dict)
            self.image_set_list.append(im_set)

    ## Work with Averaging multiple sets of data

    def get_local_hetero_averaged(self, im_sets_to_use, num_r_bins=800):
        '''Assumes that the images are already setup.'''
        #TODO: Figure out the exponential decay returned by this is valid...it's probably not

        num_r_bins=800

        # Get the maximum radius to bin, first of all
        max_r_scaled = 99999 # in mm; this is obviously ridiculous, nothing will be larger
        for im_set_index in im_sets_to_use:
            cur_im = self.image_set_list[im_set_index]
            cur_max_r = cur_im.max_radius * cur_im.get_scaling()
            if cur_max_r < max_r_scaled:
                max_r_scaled = cur_max_r

        # Set up binning
        rscaled_bins = np.linspace(0, max_r_scaled, num=num_r_bins)

        mean_list = []
        # Loop through im_sets, bin at each r
        for im_set_index in im_sets_to_use:
            cur_im = self.image_set_list[im_set_index]
            local_hetero = cur_im.get_local_hetero_df()
            local_hetero['radius_scaled'] = local_hetero['radius'] * cur_im.get_scaling()
            # Bin on the set coordinates, be alerted if binning results in NaN's (too tight)
            r_scaled_cut = pd.cut(local_hetero['radius_scaled'], rscaled_bins)
            r_scaled_groups = local_hetero.groupby(r_scaled_cut)
            cur_im_mean = r_scaled_groups.mean()
            # Check for nan's
            nan_list = cur_im_mean[cur_im_mean.isnull().any(axis=1)]
            if not nan_list.empty:
                print 'r binning is too tight; getting NaN'
                print nan_list
            cur_im_mean['im_set_index'] = im_set_index
            # Append list
            mean_list.append(cur_im_mean)
        # Combine the list of each experiment
        combined_mean_df = pd.concat(mean_list)
        # Group by the index
        result = combined_mean_df.groupby(level=0, axis=0).agg(['mean', sp.stats.sem])
        # Sort by radius scaled
        result = result.sort([('radius_scaled', 'mean')])
        # Create a column with the midpoint of each bin which is what we actually want
        result['r_scaled_midbin'] = (rscaled_bins[1:] + rscaled_bins[0:-1])/2.

        return result

    def get_nonlocal_hetero_averaged(self, im_sets_to_use, r_scaled, num_theta_bins=250):
        df_list = []
        standard_theta_bins = np.linspace(-np.pi, np.pi, num_theta_bins)

        for im_set_index in im_sets_to_use:
            cur_im_set = self.image_set_list[im_set_index]
            cur_scaling = cur_im_set.get_scaling()

            desired_r = np.around(r_scaled / cur_scaling)
            result, theta_list = cur_im_set.get_nonlocal_hetero_df_array(desired_r)
            cur_df = pd.DataFrame(data={'h':result, 'theta': theta_list})

            # Check for nan's caused by heterozygosity

            if not cur_df[cur_df.isnull().any(axis=1)].empty:
                print 'Nonlocal heterozygosity returning nan rows from im_set=' + str(im_set_index) + ', r=' + str(r_scaled)
                print cur_df[cur_df.isnull().any(axis=1)]

            cur_df['radius_scaled'] = r_scaled
            cur_df['im_set_index'] = im_set_index

            # Now bin on the standard theta bins
            theta_cut = pd.cut(cur_df['theta'], standard_theta_bins)
            cur_df = cur_df.groupby(theta_cut).mean()

            df_list.append(cur_df)

        # Now combine the df_list
        combined_df = pd.concat(df_list)
        # Groupby the index which is theta
        groups = combined_df.groupby(level=0, axis=0)
        # Get statistics
        result = groups.agg(['mean', sp.stats.sem])
        # Sort the results by theta
        result = result.sort([('theta', 'mean')])

        # Return midbin
        result['theta_midbin'] = (standard_theta_bins[1:] + standard_theta_bins[0:-1])/2.

        # Check for nan's due to binning

        if not result[result.isnull().any(axis=1)].empty:
            print 'Binning is too tight; getting NaN at r=' + str(r_scaled)
            print result[result.isnull().any(axis=1)]

        return result

class Image_Set():
    def __init__(self, image_name, path_dict):
        self.image_name = image_name
        self.path_dict = path_dict
        self.bioformats_xml = Bioformats_XML(self.path_dict['tif_folder'] + self.image_name)

        self.circle_mask = None
        self.edges_mask = None
        self.doctored_edges_mask = None
        self.channels_mask = None
        self.color_fractions = None
        self.image = None

        # Other useful stuff for data analysis
        self.image_coordinate_df = None
        self.image_coordinate_df_max_radius = None
        self.frac_df_list = None
        # Information about the maximum radius of the data we care about
        self.max_radius = None

    def finish_setup(self):
        '''This step takes a lot of time & memory but vastly speeds up future computation.'''

        # Circle mask
        try:
            self.circle_mask = ski.io.imread(self.path_dict['circle_folder'] + self.image_name, plugin='tifffile') > 0
        except IOError:
            print 'No circle mask found!'
        # Edges mask
        try:
            self.edges_mask = ski.io.imread(self.path_dict['edges_folder'] + self.image_name, plugin='tifffile') > 0
        except IOError:
            print 'No edges mask found!'
        # Doctored edges mask
        try:
            self.doctored_edges_mask = ski.io.imread(self.path_dict['doctored_edges_folder'] + self.image_name, plugin='tifffile') > 0
        except IOError:
            print 'No doctored edges mask found!'
        # Channel mask
        try:
            self.channel_masks = ski.io.imread(self.path_dict['masks_folder'] + self.image_name, plugin='tifffile') > 0
        except IOError:
            print 'No channel masks found!'

        # Find max radius in brightfield mask; needed for other functions
        self.max_radius = self.get_max_radius()

        # Based on this information, calculate fractions
        self.fractions = self.get_color_fractions()

        # Read the original image too
        self.image = ski.io.imread(self.path_dict['tif_folder'] + self.image_name, plugin='tifffile')

        # Initialize image coordinate df
        self.image_coordinate_df = self.get_image_coordinate_df()
        self.image_coordinate_df_max_radius = self.image_coordinate_df[self.image_coordinate_df['radius'] < self.max_radius]

        # Initialize channel fraction df
        self.frac_df_list = self.get_channel_frac_df()



    def get_color_fractions(self):
        sum_mask = np.zeros((self.channel_masks.shape[1], self.channel_masks.shape[2]))
        for i in range(self.channel_masks.shape[0]):
            sum_mask += self.channel_masks[i, :, :]

        # Now divide each channel by the sum
        fractions = self.channel_masks / sum_mask.astype(np.float)
        fractions[np.isnan(fractions)] = 0
        return fractions


    def get_center(self):
        '''Returns the mean center as the standard error of the mean'''
        center_list = []
        for i in range(self.circle_mask.shape[0]):
            cur_image = self.circle_mask[i, :, :]
            label_image = ski.measure.label(cur_image, neighbors=8)
            props = ski.measure.regionprops(label_image)
            for p in props:
                # There should only be one property
                center_list.append(p['centroid'])
        center_list = np.asarray(center_list)
        center_df = pd.DataFrame(data = center_list, columns=('r', 'c'))
        av_center = center_df.mean()
        std_err = center_df.apply(lambda x: sp.stats.sem(x, ddof=2))

        return av_center, std_err

    def get_max_radius(self):
        diameter_list = []
        for i in range(self.circle_mask.shape[0]):
            cur_image = self.circle_mask[i, :, :]
            # Find maximum diameter
            r, c = np.where(cur_image)
            diameter = np.float(r.max() - r.min())
            diameter_list.append(diameter)
        diameter_list = np.array(diameter_list)
        # Now find the mean radius
        max_radius = int(np.floor(diameter_list.mean()/2))
        return max_radius

    def get_image_coordinate_df(self):
        '''Returns image coordinates in r and theta. Uses the center of the brightfield mask
        as the origin.'''
        rows = np.arange(0, self.image.shape[1])
        columns = np.arange(0, self.image.shape[2])
        rmesh, cmesh = np.meshgrid(rows, columns, indexing='ij')

        r_ravel = rmesh.ravel()
        c_ravel = cmesh.ravel()

        df = pd.DataFrame(data={'r': r_ravel, 'c': c_ravel})

        av_center, std_err = self.get_center()
        df['delta_r'] = df['r'] - av_center['r']
        df['delta_c'] = df['c'] - av_center['c']
        df['radius'] = (df['delta_r']**2. + df['delta_c']**2.)**0.5
        df['theta'] = np.arctan2(df['delta_r'], df['delta_c'])

        return df

    def get_channel_frac_df(self):

        df_list = []
        for frac in self.fractions:
            df = self.image_coordinate_df.copy()
            df['f'] = frac.ravel()
            # Only keep data less than the maximum radius!
            df = df[df['radius'] < self.max_radius]

            df_list.append(df)
        return df_list

    def bin_image_coordinate_r_df(self, df):
        max_r_ceil = np.floor(df['radius'].max())
        bins = np.arange(0, max_r_ceil+ 2 , 1.5)
        groups = df.groupby(pd.cut(df.radius, bins))
        mean_groups = groups.agg(['mean'])
        return mean_groups, bins

    def bin_theta_at_r_df(self, r, delta_x=1.5):

        theta_df_list = []

        delta_theta = delta_x / float(r)
        theta_bins = np.arange(-np.pi - .5*delta_theta, np.pi + .5*delta_theta, delta_theta)

        for frac in self.frac_df_list:
            # First get the theta at the desired r; r should be an int
            theta_df = frac[(frac['radius'] >= r - delta_x/2.) & (frac['radius'] < r + delta_x/2.)]
            if not theta_df[theta_df.isnull().any(axis=1)].empty:
                print 'bin_theta_at_r_df has NaN due to r binning: r=' +str(r) + ', delta_x=' + str(delta_x), self.image_name
                print theta_df[theta_df.isnull().any(axis=1)]

            theta_cut = pd.cut(theta_df['theta'], theta_bins)
            groups = theta_df.groupby(theta_cut)
            mean_df = groups.agg(['mean'])
            # Check for nan's
            if not mean_df[mean_df.isnull().any(axis=1)].empty > 0:
                print 'theta binning in bin_theta_at_r_df is producing NaN at r=' +str(r) + ', delta_x=' + str(delta_x) + \
                      ' name= ' + self.image_name
                print mean_df[mean_df.isnull().any(axis=1)]

            theta_df_list.append(mean_df)


        return theta_df_list, theta_bins

    def delta_theta_convolve_df(self, r, delta_theta):
        '''Calculates the heterozygosity delta_theta away'''
        theta_df_list, theta_bins = self.bin_theta_at_r_df(r)
        # Now determine how many indices away you need to grab to properly do the roll
        theta_spacing = theta_bins[1] - theta_bins[0]
        theta_index = np.ceil(delta_theta / theta_spacing)
        if np.mod(theta_index, 1) == 0 and delta_theta != 0:
            theta_index -= 1

        theta_index = int(theta_index) # The number we have to roll

        conv_list = []
        for cur_theta_df in theta_df_list:
            f_values = cur_theta_df['f'].values.flatten()
            convolution = np.roll(f_values, theta_index)
            conv_list.append(convolution)

        # Return the updated lists
        new_df_list = []
        count = 0
        for cur_theta_df in theta_df_list:
            new_df = cur_theta_df.drop('f', 1)
            new_df['f', 'mean'] = conv_list[count]
            new_df_list.append(new_df)

            count += 1
        return new_df_list

    def get_local_hetero_df(self):
        local_hetero = np.zeros(self.frac_df_list[0].shape[0])

        for j in range(len(self.frac_df_list)):
            result = self.frac_df_list[j]['f']*(1-self.frac_df_list[j]['f'])
            local_hetero += result
        hetero_df = self.image_coordinate_df_max_radius.copy()
        hetero_df['h'] = local_hetero
        return hetero_df

    def get_nonlocal_hetero_df_array(self, r):
        '''Calculates the heterozygosity at every theta.'''
        theta_df_list, theta_bins = self.bin_theta_at_r_df(r)
        theta_step = theta_bins[1] - theta_bins[0]

        # Grab the desired data
        theta_f_list = []
        for cur_theta_df in theta_df_list:
            theta_f_list.append(cur_theta_df['f'].values)
        theta_f_list = np.array(theta_f_list) # [0, ...] is the first list, etc.
        convolve_list = theta_f_list.copy()

        # Number of points to calcualte
        num_points = theta_df_list[0].shape[0]

        delta_theta_list = -1.*np.ones(num_points, dtype=np.double)
        mean_h_list = -1.*np.ones(num_points, dtype=np.double)
        for i in range(num_points):
            if i == 0:
                delta_theta_list[i] = 0
            else:
                delta_theta_list[i] = delta_theta_list[i - 1] + theta_step

            # Calculate the heterozygosity
            multiplied = theta_f_list * (1 - convolve_list)
            # From multiplied, calculat the heterozygosity
            h = multiplied.sum(axis=0)
            mean_h_list[i] = h.mean()

            # Roll the convolve list by 1
            convolve_list = np.roll(convolve_list, 1, axis=1)

        # Return theta between -pi and pi
        delta_theta_list[delta_theta_list > np.pi] -= 2*np.pi

        # Now sort based on theta
        sorted_indices = np.argsort(delta_theta_list)
        delta_theta_list = delta_theta_list[sorted_indices]
        mean_h_list = mean_h_list[sorted_indices]

        return mean_h_list, delta_theta_list


    def get_local_hetero_mask(self):
        local_hetero_mask = np.zeros((self.fractions.shape[1], self.fractions.shape[2]))

        for i in range(self.fractions.shape[0]):
            for j in range(self.fractions.shape[0]):
                if i != j:
                    draw_different = self.fractions[i] * self.fractions[j]
                    local_hetero_mask += draw_different

    def get_scaling(self):
        '''Assumes x & y pixel scaling are the same'''
        scaling_str = self.bioformats_xml.pixel_nodes[0].attrib['PhysicalSizeX']
        return float(scaling_str)

    # Plotting functions
    def plot_local_hetero(self):
        '''Make a simple plot of the local heterozygosity. Useful for diagnostic purposes.'''
        hetero_df = self.get_local_hetero_df()
        binned_hetero, bins = self.bin_image_coordinate_r_df(hetero_df)
        binned_hetero['r_midbin'] = (bins[1:] + bins[:-1])/2.
        plt.semilogy(binned_hetero['r_midbin'], binned_hetero['h', 'mean'])


class Bioformats_XML():
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