__author__ = 'bryan'

import glob
import cPickle as pkl
import numpy as np

def import_files_in_folder(title, quantity, i=None, j=None, base_directory = './'):
    folder_name = title + '_' + quantity
    if (i is not None) and (j is not None):
        folder_name += '_' + str(i) + '_' + str(j)
    folder_name += '/'

    data_list = []

    files_to_import = glob.glob(base_directory + folder_name + '*.pkl')
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
                Fij_dict_list[i, j] = import_files_in_folder(title, 'Fij_sym', i=i, j=j, base_directory='./')
            else:
                Fij_dict_list[i, j] = import_files_in_folder(title, 'Fij', i=i, j=j, base_directory='./')

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