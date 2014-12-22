__author__ = 'bryan'
import skimage as ski
import skimage.io
import matplotlib.pyplot as plt
import tifffile as ti
import xml.etree.ElementTree as ET
import glob
import os

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
        return ski.io.imread(path, plugin='tifffile')

    def get_edges_mask(self, i):
        path = self.edges_folder + self.image_names[i]
        return ski.io.imread(path, plugin='tifffile')

    def get_doctored_edges_mask(self, i):
        path = self.doctored_edges_folder + self.image_names[i]
        return ski.io.imread(path, plugin='tifffile')

    def get_channels_mask(self, i):
        path = self.masks_folder + self.image_names[i]
        return ski.io.imread(path, plugin='tifffile')

    def get_image(self, i):
        path = self.tif_folder + self.image_names[i]
        return ski.io.imread(path, plugin='tifffile')

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