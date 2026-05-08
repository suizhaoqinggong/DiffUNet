
from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor
import numpy as np
import pickle
import json

data_filename = ["t2.nii.gz",
                 "flair.nii.gz",
                 "t1.nii.gz",
                 "t1ce.nii.gz"]
seg_filename = "truth.nii.gz"


class BraTSPreprocessor(MultiModalityPreprocessor):
    def read_data(self, case_name):
        data, seg_arr, properties = super().read_data(case_name)
        if seg_arr is not None:
            # BraTS raw labels: [0, 1, 2, 4] → remap to [0, 1, 2, 3]
            seg_arr[seg_arr == 4] = 3
        return data, seg_arr, properties


def process_train():
    # fullres spacing is [0.5        0.70410156 0.70410156]
    # median_shape is [602.5 516.5 516.5]
    base_dir = "/home/cjh/data/"
    image_dir = "BraTS2021"
    preprocessor = BraTSPreprocessor(base_dir=base_dir,
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = "/home/cjh/data/fullres/train/"

    preprocessor.run(output_spacing=out_spacing,
                     output_dir=output_dir,
                     all_labels=[1, 2, 3],
    )

def plan():
    base_dir = "/home/cjh/data/"
    image_dir = "BraTS2021"
    preprocessor = BraTSPreprocessor(base_dir=base_dir,
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )

    preprocessor.run_plan()


if __name__ == "__main__":
#
    plan()

    process_train()
