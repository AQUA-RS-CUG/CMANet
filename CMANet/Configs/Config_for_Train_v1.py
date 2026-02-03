# 超参数以及路径设置
class args:
    # ============= train =============
    num_class_new: int = 5

    optical_path: str = r"C:\Users\16862\Desktop\CMANet_upload\CMANet_Sample_Dataset\optical_12feature"
    sar_path: str = r"C:\Users\16862\Desktop\CMANet_upload\CMANet_Sample_Dataset\sar_4feature_5SL"
    run_information_path = r'C:\Users\16862\Desktop\CMANet_upload\RunInfo'
    model_save_path = r'C:\Users\16862\Desktop\CMANet_upload\mode_save'

    epoch = 120
    batch_size = 96
    learning_rate = 1e-4
    test_split_pro = 0.2  # 训练集
