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
        self.circle_folder = base_folder + 'circle_radius/'
        self.edges_folder = base_folder + 'edges/'
        self.doctored_edges_folder = base_folder + 'edges_doctored/'
        self.masks_folder = base_folder + 'masks/'
        self.tif_folder = base_folder + 'tif/'

        self.tif_paths = None
        self.image_names = None

        self.bioformats_xml_list = None

        self.finish_setup()

    def finish_setup(self):
        # Get a list of all images
        self.tif_paths = glob.glob(self.tif_folder + '*.ome.tif')
        self.image_names = [os.path.basename(z) for z in self.tif_paths]
        self.image_names.sort()
        # For each image, setup the xml
        self.bioformats_xml_list = []
        for cur_tif_path in self.tif_paths:
            self.bioformats_xml_list.append(Bioformats_XML(cur_tif_path))


    def get_circle_mask(self, i):
        path = self.circle_folder + self.image_names[i]
        image = ski.io.imread(path, plugin='tifffile')
        return image > 0

    def get_edges_mask(self, i):
        path = self.edges_folder + self.image_names[i]
        image = ski.io.imread(path, plugin='tifffile')
        return image > 0

    def get_doctored_edges_mask(self, i):
        path = self.doctored_edges_folder + self.image_names[i]
        image = ski.io.imread(path, plugin='tifffile')
        return image > 0

    def get_channels_mask(self, i):
        path = self.masks_folder + self.image_names[i]
        image = ski.io.imread(path, plugin='tifffile')
        return image > 0

    def get_color_fractions(self, i):
        channel_masks = self.get_channels_mask(i)
        sum_mask = np.zeros((channel_masks.shape[1], channel_masks.shape[2]))
        for i in range(channel_masks.shape[0]):
            sum_mask += channel_masks[i, :, :]

        # Now divide each channel by the sum
        fractions = channel_masks / sum_mask.astype(np.float)
        fractions[np.isnan(fractions)] = 0
        return fractions

    def get_image(self, i):
        path = self.tif_folder + self.image_names[i]
        image = ski.io.imread(path, plugin='tifffile')
        return image > 0

    def get_center(self, i):
        '''Returns the mean center as the standard error of the mean'''
        center_list = []
        circles = self.get_circle_mask(i)
        for i in range(circles.shape[0]):
            cur_image = circles[i, :, :]
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

    def get_image_coordinate_df(self, i):
        '''Returns image coordinates in r and theta. Uses the center of the brightfield mask
        as the origin.'''
        image = self.get_image(i)
        rows = np.arange(0, image.shape[1])
        columns = np.arange(0, image.shape[2])
        rmesh, cmesh = np.meshgrid(rows, columns, indexing='ij')

        r_ravel = rmesh.ravel()
        c_ravel = cmesh.ravel()

        df = pd.DataFrame(data={'r': r_ravel, 'c': c_ravel})

        av_center, std_err = self.get_center(i)
        df['delta_r'] = df['r'] - av_center['r']
        df['delta_c'] = df['c'] - av_center['c']
        df['radius'] = (df['delta_r']**2. + df['delta_c']**2.)**0.5
        df['theta'] = np.arctan2(df['delta_r'], df['delta_c'])

        return df

    def get_channel_frac_df(self, i):
        all_fractions = self.get_color_fractions(i)
        image_coordinates = self.get_image_coordinate_df(i)

        df_list = []
        for frac in all_fractions:
            df = image_coordinates.copy()
            df['f'] = frac.ravel()
            df_list.append(df)
        return df_list

    def bin_image_coordinate_r_df(self, df):
        max_r_ceil = np.ceil(df['radius'].max())
        bins = np.arange(0, max_r_ceil+ 2 , 1.5)
        groups = df.groupby(pd.cut(df.radius, bins))
        mean_groups = groups.agg(['mean'])
        return mean_groups, bins

    def bin_theta_at_r_df(self, i, r):

        theta_df_list = []

        fractions = self.get_channel_frac_df(i)

        delta_x = 1.5
        delta_theta = delta_x / float(r)
        theta_bins = np.arange(-np.pi - .01*delta_theta, np.pi + 1.01*delta_theta, delta_theta)

        for frac in fractions:
            # First get the theta at the desired r; r should be an int
            theta_df = frac[(frac['radius'] >= r - delta_x/2.) & (frac['radius'] < r + delta_x/2.)]

            theta_cut = pd.cut(theta_df['theta'], theta_bins)
            groups = theta_df.groupby(theta_cut)
            mean_df = groups.agg(['mean'])
            theta_df_list.append(mean_df)
        return theta_df_list, theta_bins

    def delta_theta_convolve_df(self, i, r, delta_theta):
        theta_df_list, theta_bins = self.bin_theta_at_r_df(i, r)
        # Now determine how many indices away you need to grab to properly do the convolution
        theta_spacing = theta_bins[1] - theta_bins[0]
        theta_index = np.ceil(delta_theta / theta_spacing)
        if np.mod(theta_index, 1) == 0:
            theta_index -= 1
        theta_index = int(theta_index)

        mid_index = theta_df_list[0].shape[0]/2

        neg_index = mid_index - theta_index
        pos_index = mid_index + theta_index

        # Create the kernel
        kernel = np.zeros(theta_df_list[0].shape[0])
        kernel[neg_index] = 0.5
        kernel[pos_index] = 0.5
        K = np.fft.fft(kernel)

        # Grab the desired f values
        conv_list = []
        for cur_theta_df in theta_df_list:
            f_values = cur_theta_df['f'].values.flatten()
            F = np.fft.fft(f_values)
            fourier_product = F*K
            convolution = np.real_if_close(np.fft.ifft(fourier_product))
            conv_list.append(convolution)

        # Return the updated lists
        new_df_list = []
        count = 0
        for cur_theta_df in theta_df_list:
            new_df = cur_theta_df.drop('f', 1)
            new_df['delta_theta_convolve'] = conv_list[count]
            new_df_list.append(new_df)

            count += 1
        return new_df_list

    def get_local_hetero_df(self, i):
        df_list = self.get_channel_frac_df(i)
        local_hetero = np.zeros(df_list[0].shape[0])

        for j in range(len(df_list)):
            result = df_list[j]['f']*(1-df_list[j]['f'])
            local_hetero += result
        hetero_df = self.get_image_coordinate_df(i)
        hetero_df['h'] = local_hetero
        return hetero_df

    def get_nonlocal_hetero_df(self, i, r, delta_theta):
        # Get DF evaluated at different points
        convolve_list = self.delta_theta_convolve(i, r, delta_theta)
        # Get local DF
        theta_df_list, theta_bins = self.bin_theta_at_r_df(i, r)

        # Calculate the heterozygosity6
        nonlocal_hetero = np.zeros(convolve_list[0].shape[0])

        for j in range(len(convolve_list)):
            result = theta_df_list[j]['f']*(1-convolve_list[j]['f'])
            nonlocal_hetero += result

        hetero_df = self.get_image_coordinate_df(i)
        hetero_df['h'] = nonlocal_hetero
        return hetero_df

    def get_local_hetero_mask(self, i):
        fractions = self.get_color_fractions(i)
        local_hetero_mask = np.zeros((fractions.shape[1], fractions.shape[2]))

        for i in range(fractions.shape[0]):
            for j in range(fractions.shape[0]):
                if i != j:
                    draw_different = fractions[i] * fractions[j]
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