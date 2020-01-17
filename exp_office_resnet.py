from __future__ import print_function
#修改
# from keras import backend as K
# K.set_image_dim_ordering('th')
#修改

import os
import numpy as np

from models.office_match_resnet import NN
from models.domain_regularizer import DomainRegularizer
from utils.office_utils import plot_activations
from utils.domain_adaption_result import calculate_adaption, calculate_print, calculate_adaption_test
from keras.preprocessing.image import ImageDataGenerator

EXP_FOLDER = 'experiments/office_resnet/'
DATASET_FOLDER = 'utils/office_dataset/'
VGG16_WEIGHTS = 'resnet_weights.h5'
N_IMAGES_AM = 2817
N_IMAGES_DSLR = 498
N_IMAGES_WC = 795
S_IMAGE = 288
S_BATCH = 2
N_REPETITIONS = 5

if __name__ == '__main__':
    """ main """

    # create folder for model parameters
    if not os.path.exists(EXP_FOLDER):
        print("\nCreating folder " + EXP_FOLDER + "...")
        os.makedirs(EXP_FOLDER)

    print("\nLoading office image data...")
    # 数据增强，并进行扩充
    datagen = ImageDataGenerator()
    am_gen = datagen.flow_from_directory(DATASET_FOLDER + 'amazon/images',
                                         target_size=(S_IMAGE, S_IMAGE),
                                         batch_size=S_BATCH)
    dslr_gen = datagen.flow_from_directory(DATASET_FOLDER + 'dslr/images',
                                           target_size=(S_IMAGE, S_IMAGE),
                                           batch_size=S_BATCH)
    wc_gen = datagen.flow_from_directory(DATASET_FOLDER + 'webcam/images',
                                         target_size=(S_IMAGE, S_IMAGE),
                                         batch_size=S_BATCH)

    print("\nCreating/Loading image representations via resnet model...")
    nn = NN(EXP_FOLDER)

    x_am, y_am = nn.create_img_repr(DATASET_FOLDER + VGG16_WEIGHTS, am_gen,
                                    'amazon', N_IMAGES_AM)
    x_wc, y_wc = nn.create_img_repr(DATASET_FOLDER + VGG16_WEIGHTS, wc_gen,
                                    'webcam', N_IMAGES_WC)
    x_dslr, y_dslr = nn.create_img_repr(DATASET_FOLDER + VGG16_WEIGHTS, dslr_gen,
                                        'dslr', N_IMAGES_DSLR)
    print("\nRandom Repetitions...")

    print("wc->dslr:")
    acc_wcdslr_None, acc_wcdslr_MMD, acc_wcdslr_MMatch, acc_wcdslr_SMD_D1, acc_wcdslr_SMD_D1_HAT, acc_wcdslr_SMD_D2, \
    acc_wcdslr_SMD_D2_HAT, acc_wcdslr_DWMD, acc_wcdslr_DWMD1, acc_wcdslr_DWMD2, acc_wcdslr_DWMD3, acc_wcdslr_DWMD4, acc_wcdslr_DWMD5, acc_wcdslr_DWMD6, acc_wcdslr_DWMD7 = calculate_adaption_test(
        EXP_FOLDER, N_REPETITIONS, NN, x_wc, y_wc, x_dslr, y_dslr)

    print("\ndslr->wc:")
    acc_dslrwc_None, acc_dslrwc_MMD, acc_dslrwc_MMatch, acc_dslrwc_SMD_D1, acc_dslrwc_SMD_D1_HAT, acc_dslrwc_SMD_D2, \
    acc_dslrwc_SMD_D2_HAT, acc_dslrwc_DWMD, acc_dslrwc_DWMD1, acc_dslrwc_DWMD2, acc_dslrwc_DWMD3, acc_dslrwc_DWMD4, acc_dslrwc_DWMD5, acc_dslrwc_DWMD6, acc_dslrwc_DWMD7 = calculate_adaption_test(
        EXP_FOLDER, N_REPETITIONS, NN, x_dslr, y_dslr, x_wc, y_wc)

    print("\nam->wc:")
    acc_amwc_None, acc_amwc_MMD, acc_amwc_MMatch, acc_amwc_SMD_D1, acc_amwc_SMD_D1_HAT, acc_amwc_SMD_D2, \
    acc_amwc_SMD_D2_HAT, acc_amwc_DWMD, acc_amwc_DWMD1, acc_amwc_DWMD2, acc_amwc_DWMD3, acc_amwc_DWMD4, acc_amwc_DWMD5, acc_amwc_DWMD6, acc_amwc_DWMD7 = calculate_adaption_test(
        EXP_FOLDER, N_REPETITIONS, NN, x_am, y_am, x_wc, y_wc)

    print("am->dslr:")
    acc_amdslr_None, acc_amdslr_MMD, acc_amdslr_MMatch, acc_amdslr_SMD_D1, acc_amdslr_SMD_D1_HAT, acc_amdslr_SMD_D2, \
    acc_amdslr_SMD_D2_HAT, acc_amdslr_DWMD, acc_amdslr_DWMD1, acc_amdslr_DWMD2, acc_amdslr_DWMD3, acc_amdslr_DWMD4, acc_amdslr_DWMD5, acc_amdslr_DWMD6, acc_amdslr_DWMD7 = calculate_adaption_test(
        EXP_FOLDER, N_REPETITIONS, NN, x_am, y_am, x_dslr, y_dslr)

    print("dslr->am:")
    acc_dslram_None, acc_dslram_MMD, acc_dslram_MMatch, acc_dslram_SMD_D1, acc_dslram_SMD_D1_HAT, acc_dslram_SMD_D2, \
    acc_dslram_SMD_D2_HAT, acc_dslram_DWMD, acc_dslram_DWMD1, acc_dslram_DWMD2, acc_dslram_DWMD3, acc_dslram_DWMD4, acc_dslram_DWMD5, acc_dslram_DWMD6, acc_dslram_DWMD7 = calculate_adaption_test(
        EXP_FOLDER, N_REPETITIONS, NN, x_dslr, y_dslr, x_am, y_am)

    print("wc->am:")
    acc_wcam_None, acc_wcam_MMD, acc_wcam_MMatch, acc_wcam_SMD_D1, acc_wcam_SMD_D1_HAT, acc_wcam_SMD_D2, \
    acc_wcam_SMD_D2_HAT, acc_wcam_DWMD, acc_wcam_DWMD1, acc_wcam_DWMD2, acc_wcam_DWMD3, acc_wcam_DWMD4, acc_wcam_DWMD5, acc_wcam_DWMD6, acc_wcam_DWMD7 = calculate_adaption_test(
        EXP_FOLDER, N_REPETITIONS, NN, x_wc, y_wc, x_am, y_am)

    print("==================================================================================")
    print("am->wc")
    calculate_print(acc_amwc_None, acc_amwc_MMD, acc_amwc_MMatch, acc_amwc_SMD_D1, acc_amwc_SMD_D1_HAT, acc_amwc_SMD_D2,
                    acc_amwc_SMD_D2_HAT, acc_amwc_DWMD, acc_amwc_DWMD1, acc_amwc_DWMD2, acc_amwc_DWMD3, acc_amwc_DWMD4,
                    acc_amwc_DWMD5, acc_amwc_DWMD6, acc_amwc_DWMD7)

    print("dslr->wc")
    calculate_print(acc_dslrwc_None, acc_dslrwc_MMD, acc_dslrwc_MMatch, acc_dslrwc_SMD_D1, acc_dslrwc_SMD_D1_HAT,
                    acc_dslrwc_SMD_D2, acc_dslrwc_SMD_D2_HAT, acc_dslrwc_DWMD, acc_dslrwc_DWMD1, acc_dslrwc_DWMD2,
                    acc_dslrwc_DWMD3, acc_dslrwc_DWMD4, acc_dslrwc_DWMD5, acc_dslrwc_DWMD6, acc_dslrwc_DWMD7)

    print("wc->dslr")
    calculate_print(acc_wcdslr_None, acc_wcdslr_MMD, acc_wcdslr_MMatch, acc_wcdslr_SMD_D1, acc_wcdslr_SMD_D1_HAT,
                    acc_wcdslr_SMD_D2, acc_wcdslr_SMD_D2_HAT, acc_wcdslr_DWMD, acc_wcdslr_DWMD1, acc_wcdslr_DWMD2,
                    acc_wcdslr_DWMD3, acc_wcdslr_DWMD4, acc_wcdslr_DWMD5, acc_wcdslr_DWMD6, acc_wcdslr_DWMD7)

    print("am->dslr")
    calculate_print(acc_amdslr_None, acc_amdslr_MMD, acc_amdslr_MMatch, acc_amdslr_SMD_D1, acc_amdslr_SMD_D1_HAT,
                    acc_amdslr_SMD_D2, acc_amdslr_SMD_D2_HAT, acc_amdslr_DWMD, acc_amdslr_DWMD1, acc_amdslr_DWMD2,
                    acc_amdslr_DWMD3, acc_amdslr_DWMD4, acc_amdslr_DWMD5, acc_amdslr_DWMD6, acc_amdslr_DWMD7)

    print("dslr->am")
    calculate_print(acc_dslram_None, acc_dslram_MMD, acc_dslram_MMatch, acc_dslram_SMD_D1, acc_dslram_SMD_D1_HAT,
                    acc_dslram_SMD_D2, acc_dslram_SMD_D2_HAT, acc_dslram_DWMD, acc_dslram_DWMD1, acc_dslram_DWMD2,
                    acc_dslram_DWMD3, acc_dslram_DWMD4, acc_dslram_DWMD5, acc_dslram_DWMD6, acc_dslram_DWMD7)

    print("wc->am")
    calculate_print(acc_wcam_None, acc_wcam_MMD, acc_wcam_MMatch, acc_wcam_SMD_D1, acc_wcam_SMD_D1_HAT, acc_wcam_SMD_D2,
                    acc_wcam_SMD_D2_HAT, acc_wcam_DWMD, acc_wcam_DWMD1, acc_wcam_DWMD2, acc_wcam_DWMD3, acc_wcam_DWMD4,
                    acc_wcam_DWMD5, acc_wcam_DWMD6, acc_wcam_DWMD7)
    print("==================================================================================")