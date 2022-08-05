import numpy as np
import os
import radiomics
import SimpleITK as sitk

from radiomics import featureextractor

def extract_features_from_images(segmented_image_path, raw_image_path):
    results = {}

    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()

    for segmented_file in os.listdir(segmented_image_path):
        raw_image = sitk.ReadImage(os.path.join(raw_image_path, segmented_file.replace('segmented_', '')),
                                   sitk.sitkUInt8)
        raw_image_arr = sitk.GetArrayFromImage(raw_image)
        segmented_image = sitk.ReadImage(os.path.join(segmented_image_path, segmented_file), 
                                         sitk.sitkUInt8)

        mask_arr = sitk.GetArrayFromImage(segmented_image)
        # mask_arr[0][0] gets the label for the mask
        featureVector = extractor.execute(raw_image, segmented_image, label=int(mask_arr[0][0]))
        results[segmented_file] = np.array([(feature, value) for feature, value in featureVector.items()], dtype=object)
    return results
