__author__ = 'bryan'
import skimage as ski
import skimage.io
import matplotlib.pyplot as plt
import tifffile as ti
import xml.etree.ElementTree as ET


class Range_Expansion_Experiment():
    def __init__(self, base_folder):
        self.circle_folder = base_folder + 'circle_radius/'
        self.edges_folder = base_folder + 'edges/'
        self.doctored_edges_folder = base_folder + 'edges_doctored/'
        self.masks_folder = base_folder + 'masks/'
        self.tif_folder = base_folder + 'tif/'

        self.xml_root = None
        self.image_nodes = None
        self.pixel_nodes = None
        self.channel_nodes = None


    def get_bioformats_xml(self, path):
        image_list = []
        xml_str = None
        with ti.TiffFile(path) as tif:
            first_page = tif.pages[0]
            for tag in first_page.tags.values():
                    if tag.name == 'image_description':
                        xml_str = tag.value
        return xml_str

    def setup_xml(self, path):
        '''Assumes there is one image for now.'''
        xml_str = self.get_bioformats_xml(path)
        self.xml_root = ET.fromstring(xml_str)
        self.image_nodes = []
        self.pixel_nodes = []
        self.channel_nodes = []
        for child in self.xml_root:
            if child.tag.endswith('Image'):
                self.image_nodes.append(child)
                for grandchild in child:
                    if grandchild.tag.endswith('Pixels'):
                        self.pixel_nodes.append(grandchild)
