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
        center_df = pd.DataFrame(data = center_list, columns=('x', 'y'))
        av_center = center_df.mean()
        std_err = center_df.apply(lambda x: sp.stats.sem(x, ddof=2))

        return av_center, std_err

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