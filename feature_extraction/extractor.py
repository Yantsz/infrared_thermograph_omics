from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt
from radiomics import featureextractor
def src_convert(src_path):
    src_img=sitk.ReadImage(src_path)
    src_array=sitk.GetArrayFromImage(src_img)
    src_array=np.squeeze(src_array)
    src_img=sitk.GetImageFromArray(src_array)
    return src_img

def mask_convert(mask_path):
    mask_image = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_image)
    mask_array = np.squeeze(mask_array)
    mask_image = sitk.GetImageFromArray(mask_array)
    return mask_image

def extract_features(extractor, src_path, mask_path):
    src = src_convert(src_path)
    mask = mask_convert(mask_path)
    features = extractor.execute(src, mask, label=1)
    return pd.DataFrame([features]).T

def read_info_file(file_path):
    with open(file_path, 'r', encoding='GBK') as info_file:
        info_txt = info_file.read()
        id, name, date = info_txt.split('_')
    return int(id), name, date


def drop_cols(df):
    drop_columns = []
    for col in df.columns:
        if col == 'label':
            df[col] = df[col].astype('category')
        else:
            try:
                df[col] = df[col].astype(np.float64)
            except:
                drop_columns.append(col)
    df.drop(drop_columns, axis=1, inplace=True)
    return df

def process_directory(directory, file_type, extractor):
    data_frames = []

    for dir0 in Path(directory).glob('*'):
        for dir in dir0.glob('*'):
            src_files = list(dir.glob(f'*{file_type}.nrrd'))
            mask_files = list(dir.glob(f'*{file_type}.seg.nrrd'))
            if src_files and mask_files:
                try:
                    df = extract_features(extractor, src_files[0], mask_files[0])
                    info_file = list(dir.glob('*.txt'))[0]
                    id, name, date = read_info_file(info_file)
                    df.rename(columns={0: id}, inplace=True)
                    data_frames.append(df)
                except Exception as e:
                    print(f"Error processing {dir}: {e}")
    return pd.concat(data_frames, axis=1)

def generate_result_xlsx(extractor, root_dir, result_type):
    root_path = Path(root_dir)
    df_normal = process_directory(root_path/'not_MS', result_type, extractor)
    df_ms = process_directory(root_path/'MS', result_type, extractor)
    df_normal_t=df_normal.T
    df_ms_t=df_ms.T
    df_normal_t['label'] = 0
    df_ms_t['label'] = 1
    df = pd.concat([df_normal_t, df_ms_t])
    df = drop_cols(df)
    return df.loc[:,'original_firstorder_10Percentile':]