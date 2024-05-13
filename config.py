import os
import datetime


class Config:
    def __init__(self):
        # prepare
        root = os.getcwd()
        self.model_save_path = os.path.join(root, 'save_model_gan')
        self.disc_save_path = os.path.join(root, 'save_model_disc')
        self.train_file_list_path = os.path.join(root, 'data_bank_voice/train_file_list_bank_linux.txt')
        self.validation_file_list_path = os.path.join(root, 'data_bank_voice/test_file_list_bank_linux.txt')
        self.val_results_file = "SEGAN-{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.resume_model = None
        # hyperparameter
        self.frame_size = 400
        self.overlap = 0.75
        self.frame_shift = int(self.frame_size * (1 - self.overlap))
        self.sr = 16000
        self.max_epochs = 200
        self.batch_size = 4
        self.d_model = 64
        self.weight_decay = 1e-7
        self.weight = [0.1, 0.9, 0.2, 0.05]
        self.history_portion = 0.2
        self.lr = 0.0004 * (0.98 ** ((-1) // 2))
        self.lr_disc = 0.0004 * (0.98 ** ((-1) // 2)) * 2

    def prepare(self):
        if not os.path.isdir(self.model_save_path):
            os.makedirs(self.model_save_path)
        if not os.path.isdir(self.disc_save_path):
            os.makedirs(self.disc_save_path)
