import os
from utils import extract_roi

input_path = '/media/data/home/xiongxiangyu/ultrasound_bening_malignant/malignent/SMP_ALL2'
output_path = '/media/data/home/xiongxiangyu/ultrasound_cropped/malignent/SPM_ALL2'


if __name__ == '__main__':
    patient_list = os.listdir(input_path)
    patient_list.sort()
    with open("patient_name.txt", "w") as external_file:
        for patient_name in patient_list:
            patient_path = os.path.join(input_path, patient_name)
            if os.path.isdir(patient_path):
                print(patient_name, file=external_file)
                print(patient_name)
                file_list = os.listdir(patient_path)

                cropped_path = os.path.join(output_path, patient_name)
                if not os.path.exists(cropped_path):
                    os.mkdir(cropped_path)
                
                for file_name in file_list:
                    if file_name.endswith(".bmp"):
                        img_path = os.path.join(patient_path, file_name)
                        save_path = os.path.join(cropped_path, file_name[:-4] + '_crop.bmp')
                        extract_roi(img_path, save_path)
    external_file.close()
