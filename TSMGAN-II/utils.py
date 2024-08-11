import numpy as np
import torch
import random
from eval_composite import SSNR
from pesq import pesq
from joblib import Parallel, delayed
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


# Email for notification
def email(title, content):
    sender = YOUR_EMAIL
    pwd = EMAIL_KEY
    receiver = YOUR_EMAIL
    try:
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['From'] = formataddr(["DDP_GPU", sender])
        msg['To'] = formataddr(["B1ubiu", receiver])
        msg['Subject'] = title
        server = smtplib.SMTP_SSL("smtp.163.com", 465)
        server.login(sender, pwd)
        server.sendmail(sender, [receiver], msg.as_string())
        server.quit()
        print("[+] 邮件发送成功")
        return True
    except Exception:
        print("[-] 邮件发送失败")
        return False


# Perform power compression on the input
def power_compress(x):
    # x[b, f_size, num_f, 2]
    real = x[..., 0]  # [ b, f_size, num_f]
    imag = x[..., 1]  # [ b, f_size, num_f]
    spec = torch.complex(real, imag)  # [ b, f_size, num_f]
    mag = torch.abs(spec)  # [ b, f_size, num_f]
    phase = torch.angle(spec)  # [ b, f_size, num_f]
    mag = mag ** 0.3  # [ b, f_size, num_f]
    real_compress = mag * torch.cos(phase)  # [ b, f_size, num_f]
    imag_compress = mag * torch.sin(phase)  # [ b, f_size, num_f]
    return torch.stack([real_compress, imag_compress], 1)  # [ b, 2, f_size, num_f]


# Perform power decompression on the input
def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1. / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)  # [b, 1, size_f, num_f, 2]


# Convert the input to a tensor
def ToTensor(x):
    return torch.from_numpy(x).float()


# Calculate the number of parameters in the model
def numParamsAll(net):
    num = 0
    for param in net.parameters():
        num += int(np.prod(param.size()))
    return num


# Calculate the number of trainable parameters in the model
def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


# Operation of the input data
def disassemble_spectrogram(spec):
    real = spec[:, 0, :, :].unsqueeze(1)
    imag = spec[:, 1, :, :].unsqueeze(1)
    mag = torch.sqrt(real ** 2 + imag ** 2)
    return real, imag, mag


# Get the label of the input data
def get_label(x):
    mean = 1 - 0.05 * x
    sigma = 0.02
    low, high = 0.95, 1
    y = torch.normal(
        mean=mean,
        std=sigma,
        size=(1, 4)
    )
    y = y.clamp(low, high)
    return y


# Calculate the loss of the self-correcting discriminator
def self_correcting_loss(discriminator, optimizer_disc, L_C, L_E, L_N):
    optimizer_disc.zero_grad()
    L_C.backward(retain_graph=True)
    grad_C_tensor = [param.grad.clone() for _, param in discriminator.named_parameters()]
    grad_C_list = torch.cat([grad.reshape(-1) for grad in grad_C_tensor], dim=0)

    optimizer_disc.zero_grad()
    L_E.backward(retain_graph=True)
    grad_E_tensor = [param.grad.clone() for _, param in discriminator.named_parameters()]
    grad_E_list = torch.cat([grad.reshape(-1) for grad in grad_E_tensor], dim=0)
    EdotE = torch.dot(grad_E_list, grad_E_list).item() + 1e-14

    optimizer_disc.zero_grad()
    L_N.backward(retain_graph=True)
    grad_N_tensor = [param.grad.clone() for _, param in discriminator.named_parameters()]
    grad_N_list = torch.cat([grad.reshape(-1) for grad in grad_N_tensor], dim=0)
    NdotN = torch.dot(grad_N_list, grad_N_list).item() + 1e-14

    CdotE = torch.dot(grad_C_list, grad_E_list).item()
    CdotN = torch.dot(grad_C_list, grad_N_list).item()
    EdotN = torch.dot(grad_E_list, grad_N_list).item()
    if CdotE > 0:
        w_C, w_E = 1, 1
        if torch.dot(w_C * grad_C_list + w_E * grad_E_list, grad_N_list).item() > 0:
            w_N = 1
        else:
            w_N = -CdotN / NdotN - EdotN / NdotN
    else:
        w_C = 1
        w_E = -CdotE / EdotE
        if torch.dot(w_C * grad_C_list + w_E * grad_E_list, grad_N_list).item() > 0:
            w_N = 1
        else:
            w_N = -CdotN / NdotN + (CdotE * EdotN) / (EdotE * NdotN)

    optimizer_disc.zero_grad()
    sc_loss = w_C * L_C + w_E * L_E + w_N * L_N
    for index, (_, param) in enumerate(discriminator.named_parameters()):
        param.grad = w_C * grad_C_tensor[index] + \
                     w_E * grad_E_tensor[index] + \
                     w_N * grad_N_tensor[index]
    return sc_loss


# Calculate the PESEQ score
def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


# Calculate the PESEQ score in batches
def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=4)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to('cuda')


# Calculate the SSNR score
def ssnr_loss(clean, noisy, sr=16000):
    try:
        snr_mean, segsnr_mean = SSNR(clean, noisy, sr)
        segSNR = np.mean(segsnr_mean)
    except:
        segSNR = -1
    return segSNR


# Calculate the SSNR score in batches
def batch_ssnr(clean, noisy):
    ssnr_score = Parallel(n_jobs=4)(delayed(ssnr_loss)(c, n) for c, n in zip(clean, noisy))
    ssnr_score = np.array(ssnr_score)
    if -1 in ssnr_score:
        return None
    ssnr_score = (ssnr_score - 5) / 20
    return torch.FloatTensor(ssnr_score).to('cuda')


# Calculate the composite score
def pesq_ssnr_score(clean, noisy, weight1=1, weight2=0):
    l = min(clean.shape[1], noisy.shape[1])
    clean = clean[:, :l]
    noisy = noisy[:, :l]
    pesq_score = batch_pesq(clean, noisy)
    # return pesq_score
    ssnr_score = batch_ssnr(clean, noisy)
    if pesq_score is None and ssnr_score is None:
        return None
    if pesq_score is None:
        print(f'[-] pesq is None')
        return ssnr_score
    if ssnr_score is None:
        return pesq_score
    result = pesq_score * weight1
    mask = ssnr_score >= 0
    result += ssnr_score * weight2 * mask
    result += pesq_score * (~mask) * weight2
    return result


# Calculate the discriminator loss
def discriminator_loss_score(clean, noisy):
    pesq_score = Parallel(n_jobs=4)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to('cuda')