import os
from utils import extract_roi

input_path = '/media/data/home/xiongxiangyu/ultrasound_benign_malignant/malignant/SPM_ALL'
output_path = '/media/data/home/xiongxiangyu/ultrasound_crop_delMark/malignant/SPM_ALL'


if __name__ == '__main__':
    patient_list = os.listdir(input_path)
    patient_list.sort()
    patient_count = 0
    figure_count = 0
    for patient_name in patient_list:
        patient_path = os.path.join(input_path, patient_name)
        if os.path.isdir(patient_path):
            print(patient_name)
            patient_count += 1
            file_list = os.listdir(patient_path)

            cropped_path = os.path.join(output_path, patient_name)
            if not os.path.exists(cropped_path):
                os.mkdir(cropped_path)
                
            for file_name in file_list:
                if file_name.endswith(".bmp"):
                    figure_count += 1
                    img_path = os.path.join(patient_path, file_name)
                    save_path = os.path.join(cropped_path, file_name[:-4] + '_crop_delMark.bmp')
                    extract_roi(img_path, save_path)
    print(patient_count)
    print(figure_count)
