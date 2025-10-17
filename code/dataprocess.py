import os
import re
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import TensorDataset, DataLoader, random_split

def extract_file_key(file_name):
    match = re.search(r'(rest|eis)_(\d+)_cap_([\d.]+)\.pkl', file_name)
    if match:
        file_type = match.group(1)
        num = int(match.group(2))
        cap = float(match.group(3))
        return (num, cap, file_type)
    return None

def train_tensor_build(directory_path, scaler_path, rest_len=14, eis_len=107):
    scaler_input = StandardScaler()
    scaler_output = StandardScaler()

    input_tensors = []
    output_tensors = []
    input_data_list = []
    output_data_list = []

    for root, dirs, files in os.walk(directory_path):
        file_pairs = {}
        for file in files:
            file_path = os.path.join(root, file)
            file_key = extract_file_key(file)
            if not file_key:
                continue
            num, cap, file_type = file_key
            key = (num, cap)
            if key not in file_pairs:
                file_pairs[key] = {'input': None, 'output': None}
            if file_type == 'rest':
                file_pairs[key]['input'] = file_path
            elif file_type == 'eis':
                file_pairs[key]['output'] = file_path

        for key in sorted(file_pairs.keys(), key=lambda x: (x[0],x[1])):
            pair = file_pairs[key]
            if not pair['input'] or not pair['output']:
                continue

            try:
                df_input = pd.read_pickle(pair['input'])
                df_input.loc[df_input['time'] < 0] = np.nan
                df_input.fillna(0, inplace=True)

                if len(df_input) < rest_len:
                    pad_rows = rest_len - len(df_input)
                    padding = pd.DataFrame(np.zeros((pad_rows, df_input.shape[1])), 
                                        columns=df_input.columns)
                    df_input = pd.concat([df_input, padding], ignore_index=True)
                elif len(df_input) > rest_len:
                    df_input = df_input.iloc[:rest_len]

                input_segment = df_input[['Ecell/V', '<I>/mA', 'Temperature/°C', 'time']].values
                input_data_list.append(input_segment)

            except Exception as e:
                continue

            try:
                df_output = pd.read_pickle(pair['output'])

                if len(df_output) < eis_len:
                    pad_rows = eis_len - len(df_output)
                    padding = pd.DataFrame(np.zeros((pad_rows, df_output.shape[1])),
                                        columns=df_output.columns)
                    df_output = pd.concat([df_output, padding], ignore_index=True)
                elif len(df_output) > eis_len:
                    df_output = df_output.iloc[:eis_len]

                if {'|Z|/Ohm', 'Phase(Z)/deg'}.issubset(df_output.columns):
                    output_segment = df_output[['|Z|/Ohm', 'Phase(Z)/deg']].iloc[1:eis_len].values
                    output_data_list.append(output_segment)
                else:
                    continue

            except Exception as e:
                continue

    if input_data_list:
        input_data_combined = np.vstack(input_data_list)
        input_data_normalized = scaler_input.fit_transform(input_data_combined)
        os.makedirs(scaler_path, exist_ok=True)
        joblib.dump(scaler_input, os.path.join(scaler_path, "scaler_input.pth"))

        split_indices = np.cumsum([len(arr) for arr in input_data_list])[:-1]
        input_data_split = np.split(input_data_normalized, split_indices)
        for segment in input_data_split:
            tensor = torch.tensor(segment, dtype=torch.float32).view(-1, 4)
            input_tensors.append(tensor)

    if output_data_list:
        output_data_combined = np.vstack(output_data_list)
        output_data_normalized = scaler_output.fit_transform(output_data_combined)
        joblib.dump(scaler_output, os.path.join(scaler_path, "scaler_output.pth"))

        split_indices = np.cumsum([len(arr) for arr in output_data_list])[:-1]
        output_data_split = np.split(output_data_normalized, split_indices)
        for segment in output_data_split:
            tensor = torch.tensor(segment, dtype=torch.float32).view(-1, 2)
            output_tensors.append(tensor)

    if input_tensors and output_tensors:
        input_tensor = torch.stack(input_tensors).permute(1, 0, 2)
        output_tensor = torch.stack(output_tensors).permute(1, 0, 2)
        return input_tensor, output_tensor
    else:
        raise ValueError("no valid data to construct tensors")

def create_dataloaders(input_tensor, output_tensor, batch_size=32, train_ratio=0.8, shuffle=True):
    input_data = input_tensor.permute(1, 0, 2)
    output_data = output_tensor.permute(1, 0, 2)
    dataset = TensorDataset(input_data, output_data)

    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = total_samples - train_size

    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader

