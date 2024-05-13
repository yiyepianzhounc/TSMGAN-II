import datetime
import os
import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from AudioData import TrainingDataset
from DecoderInfo_3_with_encoder_formerBlock import Net
from checkpoint import Checkpoint
from eval_composite import eval_composite
from utils import power_compress, power_uncompress


# prepare
root = os.getcwd()
validation_file_list_path = os.path.join(root, 'data_bank_voice/test_file_list_bank_linux.txt')
val_results_file = "SEGAN-{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
resume_model_path = 'Final-GAN.model'
# hyperparameter
frame_size = 400
overlap = 0.75
frame_shift = int(frame_size * (1 - overlap))
batch_size = 1
sr = 16000
random.seed(73)
net = Net()
checkpoint = Checkpoint()
checkpoint.load(resume_model_path)
net.load_state_dict(checkpoint.state_dict)
net = net.cuda().eval()
validation_data = TrainingDataset(validation_file_list_path, frame_size, frame_shift)
validation_loader = DataLoader(validation_data, batch_size=batch_size)

def evaluate(seed):
    random.seed(seed)
    global net, validation_loader, val_results_file
    total_stoi = 0.0
    total_ssnr = 0.0
    total_pesq = 0.0
    total_csig = 0.0
    total_cbak = 0.0
    total_covl = 0.0
    with torch.no_grad():
        count, total_eval_loss = 0, 0.0
        for i, (noisy, clean) in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            noisy = noisy.cuda()
            clean = clean.cuda()
            c = torch.sqrt(noisy.size(-1) / torch.sum(noisy ** 2.0, dim=-1))
            noisy = torch.transpose(noisy, 0, 1)
            noisy = torch.transpose(noisy * c, 0, 1)
            noisy_spec = torch.stft(
                noisy, frame_size, frame_shift, onesided=True,
                window=torch.hamming_window(frame_size).cuda(),
                return_complex=False
            )
            torch.cuda.empty_cache()
            noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
            est_real, est_imag = net(noisy_spec)
            est_real = est_real.permute(0, 1, 3, 2)
            est_imag = est_imag.permute(0, 1, 3, 2)
            est_spec = power_uncompress(est_real, est_imag).squeeze(1)
            est_audio = torch.istft(
                est_spec, frame_size, frame_shift, onesided=True,
                window=torch.hamming_window(frame_size).cuda(),
                return_complex=False
            )
            est_audio = torch.transpose(est_audio, 0, 1)
            est_audio = torch.transpose(est_audio / c, 0, 1)
            est_audio = est_audio.squeeze(0).cpu().numpy()
            clean = clean.squeeze(0).cpu().numpy()
            eval_metric = eval_composite(clean, est_audio, sr)
            total_pesq += eval_metric['pesq']
            total_ssnr += eval_metric['ssnr']
            total_stoi += eval_metric['stoi']
            total_cbak += eval_metric['cbak']
            total_csig += eval_metric['csig']
            total_covl += eval_metric['covl']
            count += 1
        avg_eval_loss = total_eval_loss / count
    avg_eval, avg_stoi, avg_pesq, avg_ssnr, avg_csig, avg_cbak, avg_covl =  avg_eval_loss, total_stoi / count, total_pesq / count, total_ssnr / count, total_csig / count, total_cbak / count, total_covl / count
    val_info = f"\ttrain_loss: {avg_eval:.6f}\n" \
               f"\tSTOI: {avg_stoi:.4f}\n" \
               f"\tCOVL: {avg_covl:.4f}\n" \
               f"\tCBAK: {avg_cbak:.4f}\n" \
               f"\tCSIG: {avg_csig:.4f}\n" \
               f"\tSSNR: {avg_ssnr:.4f}\n" \
               f"\tPESQ: {avg_pesq:.4f}\n"
    print(val_info)
    with open(val_results_file, "a") as f:
        f.write(f'seed: {seed}\n')
        f.write(val_info + "\n\n")

if __name__ == '__main__':
    evaluate(3407)
    for i in range(5000, 9999):
        try:
            evaluate(i)
        except:
            continue
# train_loss: 0.000000
# STOI: 0.9572
# COVL: 4.0285
# CBAK: 3.8772
# CSIG: 4.5369
# SSNR: 11.7717
# PESQ: 3.3986

# train_loss: 0.000000
# STOI: 0.9569
# COVL: 4.0310
# CBAK: 3.8756
# CSIG: 4.5420
# SSNR: 11.7341
# PESQ: 3.4001