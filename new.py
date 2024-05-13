import torch
import random
from pesq import pesq
from joblib import Parallel, delayed
from eval_composite import SSNR
import numpy as np


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


class ReplayBuffer():
    def __init__(self, data_size, history_portion):
        self.buffer = []
        self.buffer_size = int(data_size * history_portion)
        self.probability = int(1 // history_portion)

    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            random_index = random.randint(0, self.buffer_size - 1)
            self.buffer.pop(random_index)
        self.buffer.append(experience)

    def sample(self):
        random_index = random.randint(0, len(self.buffer) - 1)
        return self.buffer[random_index]

    def replay_or_not(self):
        return random.randint(0, self.probability) == 0


def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score


def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=4)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to('cuda')


def ssnr_loss(clean, noisy, sr=16000):
    try:
        snr_mean, segsnr_mean = SSNR(clean, noisy, sr)
        segSNR = np.mean(segsnr_mean)
    except:
        segSNR = -1
    return segSNR


def batch_ssnr(clean, noisy):
    ssnr_score = Parallel(n_jobs=4)(delayed(ssnr_loss)(c, n) for c, n in zip(clean, noisy))
    ssnr_score = np.array(ssnr_score)
    if -1 in ssnr_score:
        return None
    ssnr_score = (ssnr_score - 5) / 20
    return torch.FloatTensor(ssnr_score).to('cuda')


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


def discriminator_loss_score(clean, noisy):
    pesq_score = Parallel(n_jobs=4)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to('cuda')