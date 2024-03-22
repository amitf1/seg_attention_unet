import SimpleITK as sitk
import os
from pathlib import Path

def dicom2nifti(root_dicom_directory, nifti_output_path):
    if not os.path.exists(nifti_output_path):
        os.mkdir(nifti_output_path)
    root_dicom_directory = Path(root_dicom_directory)
    dicom_directories = root_dicom_directory.glob(os.path.join( "*", "*", "*"))
    for dicom_directory in dicom_directories:
        case_name = dicom_directory.parents[1].name
        series_reader = sitk.ImageSeriesReader()
        dicom_series = series_reader.GetGDCMSeriesIDs(dicom_directory.as_posix())

        # Loop through each series and append the images
        all_images = []
        for series_id in dicom_series:
            series_file_names = series_reader.GetGDCMSeriesFileNames(dicom_directory.as_posix(), series_id)
            if len(series_file_names) < 50:
                continue
            series_reader.SetFileNames(series_file_names)
            series_image = series_reader.Execute()
            all_images.append(series_image)

        # Concatenate images along the temporal axis if there are multiple series
        if len(all_images) > 1:
            final_image = sitk.JoinSeries(all_images)
        elif len(all_images) == 0:
            continue
        else:
            final_image = all_images[0]

        # Write the final image to NIfTI format
        sitk.WriteImage(final_image, os.path.join(nifti_output_path, case_name + ".nii.gz"))

if __name__ == "__main__":
    root_dicom = "/Users/novoha/Documents/Torso/pancreas/CT-82/manifest-1599750808610/Pancreas-CT"
    nifti_out = "/Users/novoha/Documents/Torso/pancreas/images"
    dicom2nifti(root_dicom, nifti_out)