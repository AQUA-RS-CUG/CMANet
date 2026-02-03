class args0:
    num_class: int = 5
    class_name = ['Flat', 'Sand', 'Submerged', 'Vegetation', 'Water']
    dist_border = 8
    weight_decay = 0.0001
    batch_size = 512
    data_process_path = r"C:\Users\16862\Desktop\CMANet_upload\process_file\data_process"
    data_dataset_path = r"C:\Users\16862\Desktop\CMANet_upload\process_file\data_dataset"
    run_information_path = r'C:\Users\16862\Desktop\CMANet_upload\RunInfo'


class decoder_args:
    args = {
        # Prediction result save path
        'predict_save_path': [
            r"C:\Users\16862\Desktop\CMANet_upload\CMANet_Result\patch1",
            r"C:\Users\16862\Desktop\CMANet_upload\CMANet_Result\patch2",
        ],

        # Optical input
        'optical_image_path': [
            r"C:\Users\16862\Desktop\CMANet_upload\test_image_dataset\patch_1.tif",
            r"C:\Users\16862\Desktop\CMANet_upload\test_image_dataset\patch_2.tif",
        ],

        # SAR input
        'sar_image_path': [
            r'C:\Users\16862\Desktop\CMANet_upload\test_image_dataset\patch_5SL_1',
            r'C:\Users\16862\Desktop\CMANet_upload\test_image_dataset\patch_5SL_2',
        ],

        # Model path
        'model_path': [
            r"C:\Users\16862\Desktop\CMANet_upload\mode_save\CMANet_V1.pth",
            r"C:\Users\16862\Desktop\CMANet_upload\mode_save\CMANet_V1.pth",
        ]
    }
