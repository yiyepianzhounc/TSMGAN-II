import numpy as np
import torch
from pesq import pesq
from joblib import Parallel, delayed
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr


def email(title, content):
    sender = 
    pwd = 
    receiver = 
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


def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1. / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], -1)  # [b, 1, size_f, num_f, 2]


def ToTensor(x):
    return torch.from_numpy(x).float()


def numParamsAll(net):
    num = 0
    for param in net.parameters():
        num += int(np.prod(param.size()))
    return num


def numParams(net):
    num = 0
    for param in net.parameters():
        if param.requires_grad:
            num += int(np.prod(param.size()))
    return num


def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=-1)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to('cuda')


def disassemble_spectrogram(spec):
    real = spec[:, 0, :, :].unsqueeze(1)
    imag = spec[:, 1, :, :].unsqueeze(1)
    mag = torch.sqrt(real ** 2 + imag ** 2)
    return real, imag, mag


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
