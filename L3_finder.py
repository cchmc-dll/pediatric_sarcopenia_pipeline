from pathlib import Path
from matplotlib import pyplot as plt

from L3_finder.maximum_intensity_projection import create_mip_from_path


# def main():
#     create_mip_from_path(Path("D:/muscle_segmentation/dataset_2_nifti/nifti_out/"))
#
#
# if __name__ == '__main__':
#     main()

mip = create_mip_from_path(str(Path("D:/muscle_segmentation/dataset_2_nifti/nifti_out/315.nii")))
plt.imshow(mip)
plt.show()
