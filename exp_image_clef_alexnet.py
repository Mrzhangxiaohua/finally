from __future__ import print_function
# 修改
# from keras import backend as K
# K.set_image_dim_ordering('th')
# 修改

import os
import numpy as np

from models.image_clef_mmatch_alexnet import NN
from models.domain_regularizer import DomainRegularizer
from utils.office_utils import plot_activations
from utils.domain_adaption_result import calculate_adaption, calculate_adaption_test, calculate_print

from keras.preprocessing.image import ImageDataGenerator

EXP_FOLDER = 'experiments/image_clef_alexnet/'
DATASET_FOLDER = 'utils/image_clef_dataset/'
VGG16_WEIGHTS = 'alexnet_weights.h5'
N_IMAGES_b = 600
N_IMAGES_c = 600
N_IMAGES_i = 600
N_IMAGES_p = 600

S_IMAGE = 288
S_BATCH = 2
N_REPETITIONS = 2

if __name__ == '__main__':
    """ main """

    # create folder for model parameters
    if not os.path.exists(EXP_FOLDER):
        print("\nCreating folder " + EXP_FOLDER + "...")
        os.makedirs(EXP_FOLDER)

    print("\nLoading office image data...")

    # 数据增强，并进行扩充
    datagen = ImageDataGenerator()
    b_gen = datagen.flow_from_directory(DATASET_FOLDER + 'b',
                                        target_size=(S_IMAGE, S_IMAGE),
                                        batch_size=S_BATCH)
    c_gen = datagen.flow_from_directory(DATASET_FOLDER + 'c',
                                        target_size=(S_IMAGE, S_IMAGE),
                                        batch_size=S_BATCH)
    i_gen = datagen.flow_from_directory(DATASET_FOLDER + 'i',
                                        target_size=(S_IMAGE, S_IMAGE),
                                        batch_size=S_BATCH)
    p_gen = datagen.flow_from_directory(DATASET_FOLDER + 'p',
                                        target_size=(S_IMAGE, S_IMAGE),
                                        batch_size=S_BATCH)
    print("\nCreating/Loading image representations via VGG_16 model...")

    nn = NN(EXP_FOLDER)
    # -----
    x_b, y_b = nn.create_img_repr(DATASET_FOLDER + VGG16_WEIGHTS, b_gen,
                                  'b', N_IMAGES_b)
    x_c, y_c = nn.create_img_repr(DATASET_FOLDER + VGG16_WEIGHTS, c_gen,
                                  'c', N_IMAGES_c)
    x_i, y_i = nn.create_img_repr(DATASET_FOLDER + VGG16_WEIGHTS, i_gen,
                                  'i', N_IMAGES_i)
    x_p, y_p = nn.create_img_repr(DATASET_FOLDER + VGG16_WEIGHTS, p_gen,
                                  'p', N_IMAGES_p)
    print("\nRandom Repetitions...")

    print("C->I")
    acc_ci_None, acc_ci_MMD, acc_ci_MMatch, acc_ci_SMD_D1, acc_ci_SMD_D1_HAT, acc_ci_SMD_D2, \
    acc_ci_SMD_D2_HAT, acc_ci_DWMD, acc_ci_DWMD1, acc_ci_DWMD2, acc_ci_DWMD3, acc_ci_DWMD4, acc_ci_DWMD5, acc_ci_DWMD6, acc_ci_DWMD7 = calculate_adaption_test(
        EXP_FOLDER,
        N_REPETITIONS,
        NN, x_c, y_c,
        x_i, y_i)

    print("C->P")
    acc_cp_None, acc_cp_MMD, acc_cp_MMatch, acc_cp_SMD_D1, acc_cp_SMD_D1_HAT, acc_cp_SMD_D2, \
    acc_cp_SMD_D2_HAT, acc_cp_DWMD, acc_cp_DWMD1, acc_cp_DWMD2, acc_cp_DWMD3, acc_cp_DWMD4, acc_cp_DWMD5, acc_cp_DWMD6, acc_cp_DWMD7 = calculate_adaption_test(
        EXP_FOLDER,
        N_REPETITIONS,
        NN, x_c, y_c,
        x_p, y_p)
    print("I->C")
    acc_ic_None, acc_ic_MMD, acc_ic_MMatch, acc_ic_SMD_D1, acc_ic_SMD_D1_HAT, acc_ic_SMD_D2, \
    acc_ic_SMD_D2_HAT, acc_ic_DWMD, acc_ic_DWMD1, acc_ic_DWMD2, acc_ic_DWMD3, acc_ic_DWMD4, acc_ic_DWMD5, acc_ic_DWMD6, acc_ic_DWMD7 = calculate_adaption_test(
        EXP_FOLDER,
        N_REPETITIONS,
        NN, x_i, y_i,
        x_c, y_c)

    print("I->P")
    acc_ip_None, acc_ip_MMD, acc_ip_MMatch, acc_ip_SMD_D1, acc_ip_SMD_D1_HAT, acc_ip_SMD_D2, \
    acc_ip_SMD_D2_HAT, acc_ip_DWMD, acc_ip_DWMD1, acc_ip_DWMD2, acc_ip_DWMD3, acc_ip_DWMD4, acc_ip_DWMD5, acc_ip_DWMD6, acc_ip_DWMD7 = calculate_adaption_test(
        EXP_FOLDER,
        N_REPETITIONS,
        NN, x_i, y_i,
        x_p, y_p)
    print("P->C")
    acc_pc_None, acc_pc_MMD, acc_pc_MMatch, acc_pc_SMD_D1, acc_pc_SMD_D1_HAT, acc_pc_SMD_D2, \
    acc_pc_SMD_D2_HAT, acc_pc_DWMD, acc_pc_DWMD1, acc_pc_DWMD2, acc_pc_DWMD3, acc_pc_DWMD4, acc_pc_DWMD5, acc_pc_DWMD6, acc_pc_DWMD7 = calculate_adaption_test(
        EXP_FOLDER,
        N_REPETITIONS,
        NN, x_p, y_p,
        x_c, y_c)
    print("P->I")
    acc_pi_None, acc_pi_MMD, acc_pi_MMatch, acc_pi_SMD_D1, acc_pi_SMD_D1_HAT, acc_pi_SMD_D2, \
    acc_pi_SMD_D2_HAT, acc_pi_DWMD, acc_pi_DWMD1, acc_pi_DWMD2, acc_pi_DWMD3, acc_pi_DWMD4, acc_pi_DWMD5, acc_pi_DWMD6, acc_pi_DWMD7 = calculate_adaption_test(
        EXP_FOLDER,
        N_REPETITIONS,
        NN, x_p, y_p,
        x_i, y_i)

    print("==================================================================================")
    print("C->I")
    calculate_print(acc_ci_None, acc_ci_MMD, acc_ci_MMatch, acc_ci_SMD_D1, acc_ci_SMD_D1_HAT, acc_ci_SMD_D2,
                    acc_ci_SMD_D2_HAT, acc_ci_DWMD, acc_ci_DWMD1, acc_ci_DWMD2, acc_ci_DWMD3, acc_ci_DWMD4,
                    acc_ci_DWMD5, acc_ci_DWMD6, acc_ci_DWMD7)

    print("C->P")
    calculate_print(acc_cp_None, acc_cp_MMD, acc_cp_MMatch, acc_cp_SMD_D1, acc_cp_SMD_D1_HAT, acc_cp_SMD_D2,
                    acc_cp_SMD_D2_HAT, acc_cp_DWMD, acc_cp_DWMD1, acc_cp_DWMD2, acc_cp_DWMD3, acc_cp_DWMD4,
                    acc_cp_DWMD5, acc_cp_DWMD6, acc_cp_DWMD7)

    print("I->C")
    calculate_print(acc_ic_None, acc_ic_MMD, acc_ic_MMatch, acc_ic_SMD_D1, acc_ic_SMD_D1_HAT, acc_ic_SMD_D2,
                    acc_ic_SMD_D2_HAT, acc_ic_DWMD, acc_ic_DWMD1, acc_ic_DWMD2, acc_ic_DWMD3, acc_ic_DWMD4,
                    acc_ic_DWMD5, acc_ic_DWMD6, acc_ic_DWMD7)

    print("I->P")
    calculate_print(acc_ip_None, acc_ip_MMD, acc_ip_MMatch, acc_ip_SMD_D1, acc_ip_SMD_D1_HAT, acc_ip_SMD_D2,
                    acc_ip_SMD_D2_HAT, acc_ip_DWMD, acc_ip_DWMD1, acc_ip_DWMD2, acc_ip_DWMD3, acc_ip_DWMD4,
                    acc_ip_DWMD5, acc_ip_DWMD6, acc_ip_DWMD7)

    print("P->C")
    calculate_print(acc_pc_None, acc_pc_MMD, acc_pc_MMatch, acc_pc_SMD_D1, acc_pc_SMD_D1_HAT, acc_pc_SMD_D2,
                    acc_pc_SMD_D2_HAT, acc_pc_DWMD, acc_pc_DWMD1, acc_pc_DWMD2, acc_pc_DWMD3, acc_pc_DWMD4,
                    acc_pc_DWMD5, acc_pc_DWMD6, acc_pc_DWMD7)

    print("P->I")
    calculate_print(acc_pi_None, acc_pi_MMD, acc_pi_MMatch, acc_pi_SMD_D1, acc_pi_SMD_D1_HAT, acc_pi_SMD_D2,
                    acc_pi_SMD_D2_HAT, acc_pi_DWMD, acc_pi_DWMD1, acc_pi_DWMD2, acc_pi_DWMD3, acc_pi_DWMD4,
                    acc_pi_DWMD5, acc_pi_DWMD6, acc_pi_DWMD7)
    print("==================================================================================")