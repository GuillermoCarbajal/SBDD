#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:34:43 2018

@author: carbajal
"""

import os
import argparse

def prepare_data(source_folder, dst_folder):
    splits = ['train', 'test']

    for split in splits:
        setDirName = os.path.join(source_folder, split)
        for (rootDir, dirNames, filenames) in os.walk(setDirName):
            print('rootDir=' , rootDir )
            print('dirNames=' , dirNames)
            for directorio in dirNames:
                #palabras = directorio.split(' ')
                if directorio == 'blur' or directorio == 'blur_gamma' or directorio == 'sharp':
                    sequenceName = rootDir.split('/')[-1]
                    print('Processing sequence', sequenceName)
                    imagesDir = os.path.join(rootDir, directorio)
                    imagenames = os.listdir(imagesDir)
                    for imgname in imagenames:
                        #print('original_image_name', imgname)
                        outputImageName = sequenceName + '_' + imgname
                        #print('output_image_name', outputImageName)

                        outputDir = os.path.join(dst_folder, directorio, split)
                        if not os.path.isdir(outputDir):
                            os.makedirs(outputDir)

                        symbolicName = os.path.join(outputDir, outputImageName )
                        if os.path.isfile(symbolicName):
                            os.unlink(symbolicName)
                        originalImage = os.path.join(imagesDir, imgname)
                        print('original image ', originalImage)
                        print('symbolic name ', symbolicName)
                        try:
                            os.symlink(originalImage, symbolicName)
                        except:
                            print("Something happened")
                            continue
    return
                       
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--GOPRO_original_data_folder', type=str, help='GoPro original folder', required=True,
                        default='/media/carbajal/OS/data/datasets/GOPRO_Large/original-data')
    parser.add_argument('--prepared_data_folder', type=str, help='Prepared data folder', required=True,
                        default='prepared-data')
                        
    args = parser.parse_args()
    
    prepare_data(args.GOPRO_original_data_folder, args.prepared_data_folder)
