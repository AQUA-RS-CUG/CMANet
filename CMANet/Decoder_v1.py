import os
import time
import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from Configs.Config_for_Decoder_v1 import decoder_args, args0
from torch.utils.data import Dataset, DataLoader
from model.CMANet_model import CMANet
from osgeo import gdal
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['PROJ_LIB'] = r'D:\APP\Anaconda\envs\sdsd_torch\Lib\site-packages\osgeo\data\proj'
os.environ['GDAL_DATA'] = r'D:\APP\Anaconda\envs\sdsd_torch\Lib\site-packages\osgeo\data'
gdal.UseExceptions()  # 启用异常处理


# 新模型 SVTransNet 测试程序代码（单幅图像测试；5类迁移后样本）


class Mydataset(Dataset):
    def __init__(self, optical_patch_list, sar_patch_list, normalize=True):
        super(Mydataset, self).__init__()
        # assert len(optical_patch_list) == len(sar_patch_list)
        self.optical_image_list = optical_patch_list
        self.sar_image_list = sar_patch_list
        self.normalize = normalize

    def __getitem__(self, index):
        optical_image = self.optical_image_list[index]
        sar_image = self.sar_image_list[index]
        if self.normalize:  # Normalize optical_image and sar_image here
            optical_image = self.normalize_image(optical_image)
            sar_image = self.normalize_image(sar_image)
        optical_image = torch.from_numpy(np.array(optical_image).astype(np.float32))
        sar_image = torch.from_numpy(np.array(sar_image).astype(np.float32))
        sample = {
            'optical_image': optical_image,
            'sar_image': sar_image
        }
        return sample

    def __len__(self):
        return len(self.optical_image_list)

    def normalize_image(self, image):
        image_min = np.min(image)
        image_max = np.max(image)
        normalized_image = (image - image_min) / (image_max - image_min)
        return normalized_image


def crop_patches_any_size(optical_image, sar_image, edge_num):
    row_num = optical_image.shape[1]
    col_num = optical_image.shape[2]
    sar_patches, optical_patches, patches_index = [], [], []
    for row in range(edge_num):
        for col in range(edge_num):
            x1 = int((row / edge_num) * row_num)
            x2 = int(((row + 1) / edge_num) * row_num + args0.dist_border * 2)
            y1 = int((col / edge_num) * col_num)
            y2 = int(((col + 1) / edge_num) * col_num + args0.dist_border * 2)
            optical_patches.append(optical_image[:, x1:x2, y1:y2])
            sar_patches.append(sar_image[:, :, x1:x2, y1:y2])
            patches_index.append(((x1, x2), (y1, y2)))
    return optical_patches, sar_patches, patches_index, row_num, col_num


def data_process(optical_image_path, timeseries_image_path_list, data_process_path):
    print("==================== Data saved as separate blocks ====================")
    start_time_process = time.time()
    # 读取图像
    sar_image_list = []
    for index in range(len(timeseries_image_path_list)):
        sar_input_image_path = timeseries_image_path_list[index]
        sar_image = tiff.imread(sar_input_image_path).astype(np.float32)
        sar_image_pad = np.zeros(shape=(
            sar_image.shape[0], sar_image.shape[1] + 2 * args0.dist_border, sar_image.shape[2] + 2 * args0.dist_border))
        sar_image_pad[:, args0.dist_border:sar_image_pad.shape[1] - args0.dist_border,
        args0.dist_border:sar_image_pad.shape[2] - args0.dist_border] = sar_image
        del sar_image
        sar_image_list.append(sar_image_pad)
        del sar_image_pad
    sar_image = np.array(sar_image_list)
    del sar_image_list
    optical_image = tiff.imread(optical_image_path).astype(np.float32)
    optical_image_pad = np.zeros(shape=(
        optical_image.shape[0], optical_image.shape[1] + 2 * args0.dist_border,
        optical_image.shape[2] + 2 * args0.dist_border))
    optical_image_pad[:, args0.dist_border:optical_image_pad.shape[1] - args0.dist_border,
    args0.dist_border:optical_image_pad.shape[2] - args0.dist_border] = optical_image
    optical_image = optical_image_pad
    del optical_image_pad

    optical_patches, sar_patches, patches_index, row_num, col_num = crop_patches_any_size(optical_image=optical_image,
                                                                                          sar_image=sar_image,
                                                                                          edge_num=5)

    for i in range(len(optical_patches)):
        tiff.imwrite(data_process_path + r'\optical_patches_{}.tif'.format(i + 1), optical_patches[i])
        tiff.imwrite(data_process_path + r'\sar_patches_{}.tif'.format(i + 1), sar_patches[i])
        np.save(data_process_path + r'\patch_index_{}.npy'.format(i + 1), patches_index[i])

    end_time_process = time.time()
    print("Data processing save time is: {:.4f}s".format(end_time_process - start_time_process))
    print("\n")
    return row_num, col_num


def decoder(optical_image_list, sar_image_list, model, device):

    model.to(device)
    model.eval()
    dataset = Mydataset(optical_patch_list=optical_image_list, sar_patch_list=sar_image_list)
    predict_dataloader = DataLoader(dataset=dataset, batch_size=args0.batch_size, shuffle=False, num_workers=0,
                                    drop_last=False)
    pred_list_output = []
    with torch.no_grad():
        for i, batch in enumerate(predict_dataloader, 0):
            optical_image = batch['optical_image'].to(device)
            sar_image = batch['sar_image'].to(device)
            out_level3 = model(optical_image, sar_image)
            probas_output = F.softmax(out_level3, dim=1)
            _, pred_output = torch.max(probas_output, dim=1)
            pred_output = pred_output.cpu().numpy()
            pred_list_output.extend(pred_output)
    return pred_list_output


def patches_save_predict(optical_patch_paths, sar_patch_paths, patch_index_path, model, device, predict_save_path,
                         row_num, col_num, image_num):
    all_start_time = time.time()
    predict_image = np.zeros(shape=(row_num - 2 * args0.dist_border, col_num - 2 * args0.dist_border))
    predict_image_o2 = np.zeros(shape=(row_num - 2 * args0.dist_border, col_num - 2 * args0.dist_border))
    predict_image_o3 = np.zeros(shape=(row_num - 2 * args0.dist_border, col_num - 2 * args0.dist_border))
    predict_image_s3 = np.zeros(shape=(row_num - 2 * args0.dist_border, col_num - 2 * args0.dist_border))
    for i in range(len(optical_patch_paths)):
        print("======= Obtain patch {} for processing =======".format(i + 1))
        start_time = time.time()

        optical_image = tiff.imread(optical_patch_paths[i])
        sar_image = tiff.imread(sar_patch_paths[i])
        patch_index = np.load(patch_index_path[i])
        # assert optical_image.shape[1] == sar_image.shape[2]
        # assert optical_image.shape[2] == sar_image.shape[3]
        patch_row = optical_image.shape[1]
        patch_col = optical_image.shape[2]
        predict_patch = np.zeros(shape=(patch_row - 2 * args0.dist_border, patch_col - 2 * args0.dist_border))
        predict_patch_o2 = np.zeros(shape=(patch_row - 2 * args0.dist_border, patch_col - 2 * args0.dist_border))
        predict_patch_o3 = np.zeros(shape=(patch_row - 2 * args0.dist_border, patch_col - 2 * args0.dist_border))
        predict_patch_s3 = np.zeros(shape=(patch_row - 2 * args0.dist_border, patch_col - 2 * args0.dist_border))
        optical_patches, sar_patches = [], []
        for row in range(optical_image.shape[1] - 2 * args0.dist_border):
            row += args0.dist_border
            for col in range(optical_image.shape[2] - 2 * args0.dist_border):
                col += args0.dist_border
                optical_patch = optical_image[:, row - args0.dist_border:row + args0.dist_border,
                                col - args0.dist_border:col + args0.dist_border]
                sar_patch = sar_image[:, :, row - args0.dist_border:row + args0.dist_border,
                            col - args0.dist_border:col + args0.dist_border]
                optical_patches.append(optical_patch)
                sar_patches.append(sar_patch)

        pred_list_output = decoder(optical_patches, sar_patches, model, device)

        pred_list_output = np.array(pred_list_output).reshape(patch_row - 2 * args0.dist_border,
                                                              patch_col - 2 * args0.dist_border)
        predict_patch = pred_list_output

        predict_image[patch_index[0][0]:patch_index[0][1] - 2 * args0.dist_border,
        patch_index[1][0]:patch_index[1][1] - 2 * args0.dist_border] = predict_patch
        end_time_1 = time.time()
        print("Predicting the time of block {} is: {:.4f}s".format(i + 1, end_time_1 - start_time))
    tiff.imwrite(
        predict_save_path + r'\CMANet_V1_output{}.tif'.format(image_num),
        predict_image)
    end_time_2 = time.time()
    print("Total forecast time is: {:.4f}s".format(end_time_2 - all_start_time))
    print("\n")


if __name__ == "__main__":
    args = decoder_args.args
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time_all = time.time()
    epoch = 2
    time_list = []
    for i in range(epoch):
        for_start_time = time.time()
        print("========== Start {} th model test ==========".format(i + 1))
        # 获取参数
        predict_save_path = args['predict_save_path'][i]
        optical_image_path = args['optical_image_path'][i]
        sar_image_path = args['sar_image_path'][i]
        model_path = args['model_path'][i]
        data_process_path = args0.data_process_path
        data_dataset_path = args0.data_dataset_path
        num_class = args0.num_class

        sar_image_path_list = [os.path.join(sar_image_path, item) for item in os.listdir(sar_image_path)]
        timeseries_image_path_list = []
        for image_path in sar_image_path_list:
            image_path_name, image_path_type = os.path.splitext(image_path)
            if image_path_type == r'.tif':
                timeseries_image_path_list.append(image_path)

        model = CMANet(in_channel1=12, in_channel2=4, num_class=num_class, hidden_dim=1024, num_heads=8, dropout=0.1)
        model.load_state_dict(torch.load(model_path))

        row_num, col_num = data_process(optical_image_path, timeseries_image_path_list, data_process_path)
        patch_paths = [os.path.join(data_process_path, item) for item in os.listdir(data_process_path)]
        optical_patch_paths, sar_patch_paths, patch_index_path = [], [], []
        for path in patch_paths:
            if "sar" in path:
                sar_patch_paths.append(path)
            if "optical" in path:
                optical_patch_paths.append(path)
            if "index" in path:
                patch_index_path.append(path)

        patches_save_predict(optical_patch_paths, sar_patch_paths, patch_index_path, model, device, predict_save_path,
                             row_num, col_num, image_num=i + 1)
        path_list = [os.path.join(data_process_path, item) for item in os.listdir(data_process_path)]
        for i in range(len(os.listdir(data_process_path))):
            if os.path.isfile(path_list[i]):
                os.remove(path_list[i])
            if os.path.isdir(path_list[i]):
                os.rmdir(path_list[i])
        for_end_time = time.time()
        time_list.append(for_end_time - for_start_time)

    np.save(args0.run_information_path + r'\CMANet_V1.npy', time_list)
    end_time_all = time.time()
    run_time = end_time_all - start_time_all
    hour = run_time // 3600
    minute = (run_time - 3600 * hour) // 60
    second = run_time - 3600 * hour - 60 * minute
    print(f'Test program runtime: {hour}小时{minute}分钟{second}秒')
    print("\n")
