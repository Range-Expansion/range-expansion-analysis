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
        self.frac_df_list = None

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

        # Based on this information, calculate fractions
        self.fractions = self.get_color_fractions()

        # Read the original image too
        self.image = ski.io.imread(self.path_dict['tif_folder'] + self.image_name, plugin='tifffile')

        # Initialize image coordinate df
        self.image_coordinate_df = self.get_image_coordinate_df()

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
            df_list.append(df)
        return df_list

    def bin_image_coordinate_r_df(self, df):
        max_r_ceil = np.ceil(df['radius'].max())
        bins = np.arange(0, max_r_ceil+ 2 , 1.5)
        groups = df.groupby(pd.cut(df.radius, bins))
        mean_groups = groups.agg(['mean'])
        return mean_groups, bins

    def bin_theta_at_r_df(self, r):

        theta_df_list = []

        delta_x = 1.5
        delta_theta = delta_x / float(r)
        theta_bins = np.arange(-np.pi - .01*delta_theta, np.pi + 1.01*delta_theta, delta_theta)

        for frac in self.frac_df_list:
            # First get the theta at the desired r; r should be an int
            theta_df = frac[(frac['radius'] >= r - delta_x/2.) & (frac['radius'] < r + delta_x/2.)]

            theta_cut = pd.cut(theta_df['theta'], theta_bins)
            groups = theta_df.groupby(theta_cut)
            mean_df = groups.agg(['mean'])
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
        hetero_df = self.image_coordinate_df.copy()
        hetero_df['h'] = local_hetero
        return hetero_df

    def get_nonlocal_hetero_df(self, r, delta_theta):
        # Get DF evaluated at different points
        convolve_list = self.delta_theta_convolve_df(r, delta_theta)
        # Get local DF
        theta_df_list, theta_bins = self.bin_theta_at_r_df(r)

        # Calculate the heterozygosity
        nonlocal_hetero = np.zeros(convolve_list[0].shape[0])

        for j in range(len(convolve_list)):
            result = theta_df_list[j]['f']*(1-convolve_list[j]['f'])
            nonlocal_hetero += result.values.flatten()

        hetero_df = theta_df_list[0].drop('f', 1)
        hetero_df['h', 'mean'] = nonlocal_hetero
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