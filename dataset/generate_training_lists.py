import os 
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', '-dd', type=str, help='dataset folder', required=False,
                        default='SBDD')
    parser.add_argument('--output_list', '-ol', type=str, help='output list', required=False,
                        default='training_list.txt')

    opt = parser.parse_args()
    sets_folders = os.listdir(opt.dataset_dir)
    print(sets_folders)
    with open(opt.output_list, 'w+') as f:
    
        for folder in sets_folders:
            blurry_folder = os.path.join(opt.dataset_dir,folder,'blurry')
            sharp_folder = os.path.join(opt.dataset_dir,folder,'sharp')
            print(blurry_folder, sharp_folder)
            blurry_images = os.listdir(blurry_folder)
            sharp_images = os.listdir(sharp_folder)
            blurry_images.sort()
            sharp_images.sort()
            for blurry_img, sharp_img in zip(blurry_images, sharp_images):
                f.write(os.path.join(sharp_folder,sharp_img) + ' ' + os.path.join(blurry_folder, blurry_img)  + '\n')
    f.close()
