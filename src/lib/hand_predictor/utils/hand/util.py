import numpy as np
import pandas as pd
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from utils.hand.model import (BothHandCNN, BothHandFullConnected,
                              BothHandLRchannelGRU, BothHandTransformer,
                              HandConvNet, HandConvNet_o,
                              HandMultichannelCNNGRU, HandRNNConvNet,
                              SampleCNNGA)


def print_group_ratio(filename_label_df: pd.DataFrame,
                      dataset_label_list: list = None,
                      title: str = None):
    '''
    filename_label_df: table of the filename and label map
        [0]: csv_name (include PUN and Study Number, e.g. 'P001_2020-01-01-1' or 'P001_2020-01-01-1_1_2.csv')
        [1]: label
    datset_label_list: all your data (with your label)
    title: print title
    '''
    # rename colname
    filename_label_df_annot = filename_label_df[[0, 1]]
    filename_label_df_annot.columns = ["csv_name", "label"]

    # split PUN to single column
    filename_label_df_annot["PUN"] = [
        each[0] for each in filename_label_df_annot["csv_name"].str.split("_")
    ]
    filename_label_df_annot["Study Number"] = [
        each[1] for each in filename_label_df_annot["csv_name"].str.split("_")
    ]

    labels = filename_label_df_annot["label"]
    labels = np.array(labels)
    label_names, counts = np.unique(labels, return_counts=True)

    if title:
        print("-" * 70)
        print(title)

    for label_name, count in zip(label_names, counts):
        filename_label_df_sub = filename_label_df_annot[
            filename_label_df_annot["label"] == label_name]
        num_of_pt = len(filename_label_df_sub["PUN"].unique())
        num_of_trial = len(filename_label_df_sub["Study Number"].unique())

        if dataset_label_list is not None:
            num_of_data = sum(each == label_name
                              for each in dataset_label_list)
            print(
                f"{label_name}: {num_of_data} via data-balanced method (from {count} records, {num_of_trial} trials, {num_of_pt} people) || {num_of_data/len(dataset_label_list)*100:.2f}%"
            )
        else:
            print(
                f"{label_name}: {count} records (from {num_of_trial} trials, {num_of_pt} people) || {count/len(labels)*100:.2f}%"
            )

    print("-" * 70)

    return None


###


def parse_args_keypoint(keypoint_string):
    keypoint_string = keypoint_string.replace(" ", "")
    kpt_no = keypoint_string.split("-")[0]
    kpt_axis = keypoint_string.split("-")[1]
    assert kpt_axis in ("xyz", "xy", "yz", "xz", "x", "y", "z")

    if kpt_no == "all":
        keypoint_list = list(range(21))
    else:
        keypoint_list = [int(item) for item in kpt_no.split(',')]

    all_channels = ["timestamp"]  # first column
    for axis in kpt_axis:
        all_channels = all_channels + [
            f"{axis}_{item}" for item in keypoint_list
        ]
    in_channels = len(all_channels)

    return all_channels, in_channels


###


def get_model_by_name(model_name, crop_len=None, n_classes=2, in_channels=42, device='cuda:0'):
    if model_name == "sampleCNNGA":
        model = SampleCNNGA(n_classes=n_classes, in_channels=in_channels)
    elif model_name == "handconvnet":
        model = HandConvNet(input_channels=in_channels,
                            output_class=n_classes,
                            crop_len=crop_len,
                            device=device)
    elif model_name == "handconvnet_o":
        model = HandConvNet_o(input_channels=in_channels,
                              output_class=n_classes,
                              crop_len=crop_len,
                              device=device)
    elif model_name == "handrnnconvnet":
        model = HandRNNConvNet(input_channels=in_channels,
                               output_class=n_classes,
                               crop_len=crop_len,
                               device=device)
    elif model_name == "handmultichannelconvgrunet":
        model = HandMultichannelCNNGRU(input_channels=in_channels,
                                       output_class=n_classes,
                                       crop_len=crop_len,
                                       device=device)
    elif model_name == "bothhandFC":
        model = BothHandFullConnected(output_class=n_classes, in_channels=in_channels)
    elif model_name == "bothhandTransformer":
        model = BothHandTransformer(output_class=n_classes, in_channels=in_channels)
    elif model_name == "bothhandCNN":
        model = BothHandCNN(output_class=n_classes, in_channels=in_channels)
    elif model_name == "bothhandLRChGRU":
        model = BothHandLRchannelGRU(output_class=n_classes, in_channels=in_channels)
    else:
        raise NotImplementedError

    return model


###


def get_scheduler_by_name(scheduler_name, optimizer, num_epochs):
    if scheduler_name == "StepLR":
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=num_epochs // 5,
                                        gamma=0.5)

    elif scheduler_name == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=5,
                                                   # T_max=num_epochs // 10,
                                                   )

    elif scheduler_name == "LambdaLR":
        lambda_func = lambda epoch: (0.95**epoch) * 0.5 * (1 + np.cos(np.pi + np.pi * epoch / 5))
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

    else:
        raise NotImplementedError

    return scheduler


###


def get_optimizer_by_name(optimizer_name, model, lr, weight_decay):
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              weight_decay=weight_decay,
                              momentum=0.9)

    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=lr,
                               betas=(0.9, 0.999),
                               eps=1e-08,
                               weight_decay=weight_decay,
                               amsgrad=False)

    else:
        raise NotImplementedError

    return optimizer
