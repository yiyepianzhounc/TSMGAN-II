import os
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distributed

from AudioData import TrainingDataset
from generator import Net
from discriminator import Discriminator
from checkpoint import Checkpoint
from eval_composite import eval_composite
from new import self_correcting_loss, pesq_ssnr_score
from utils import numParamsAll, power_compress, power_uncompress, email
from config import Config

config = Config()


class GAN_Trainer:
    def __init__(self, model, model_disc, device, is_load_epoch=None):
        config.prepare()
        self.device = device
        self.gen = model.to(self.device)
        self.disc = model_disc.to(self.device)
        self.batch_size = config.batch_size
        self.max_epochs = config.max_epochs
        self.frame_size = config.frame_size
        self.frame_shift = config.frame_shift
        self.model_save_path = config.model_save_path
        self.disc_save_path = config.disc_save_path
        self.weight = config.weight
        self.val_results_file = config.val_results_file
        self.sr = config.sr
        self.epoch = 0
        if is_load_epoch is not None and self.device == 0:
            self.load_checkpoint(is_load_epoch)
        self.gen = DDP(self.gen, device_ids=[self.device], output_device=self.device, find_unused_parameters=True)
        self.disc = DDP(self.disc, device_ids=[self.device], output_device=self.device, find_unused_parameters=True)
        train_data = TrainingDataset(
            config.train_file_list_path,
            self.frame_size, self.frame_shift
        )
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=DistributedSampler(train_data)
        )
        validation_data = TrainingDataset(
            config.validation_file_list_path,
            self.frame_size, self.frame_shift
        )
        self.validation_loader = DataLoader(
            validation_data,
            batch_size=self.batch_size,
            sampler=DistributedSampler(validation_data)
        )
        self.gen_optimizer = torch.optim.Adam(
            self.gen.parameters(), lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.disc_optimizer = torch.optim.Adam(
            self.disc.parameters(), lr=config.lr_disc,
            weight_decay=config.weight_decay
        )
        self.lr = config.lr
        self.lr_disc = config.lr_disc
        self.best_val_loss = float("-inf")
        self.now_val_loss = float("-inf")
        if self.device == 0:
            print(f'[+] Number of all parameters: {numParamsAll(model)}')

    def load_checkpoint(self, is_load_epoch):
        checkpoint = Checkpoint()
        checkpoint.load(self.model_save_path + f"/latest_model-{is_load_epoch}.model")
        self.epoch = checkpoint.start_epoch + 1
        self.gen.load_state_dict(checkpoint.state_dict)
        # self.gen_optimizer.load_state_dict(checkpoint.optimizer)
        self.best_val_loss = checkpoint.best_val_loss
        checkpoint.load(self.disc_save_path + f"/latest_model-{is_load_epoch}.model")
        self.disc.load_state_dict(checkpoint.state_dict)
        # self.disc_optimizer.load_state_dict(checkpoint.optimizer)

    def save_checkpoint(self, epoch):
        is_best = False
        if self.now_val_loss < self.best_val_loss:
            self.best_val_loss = self.now_val_loss
            is_best = True
        checkpoint = Checkpoint(
            start_epoch=epoch,
            state_dict=self.gen.module.state_dict(),
            optimizer=self.gen_optimizer.state_dict()
        )
        checkpoint.save(
            is_best=is_best,
            filename=self.model_save_path + f"/latest_model-{epoch}.model",
            best_model=self.model_save_path + f"/best_model.model"
        )
        checkpoint = Checkpoint(
            start_epoch=epoch,
            state_dict=self.disc.module.state_dict(),
            optimizer=self.disc_optimizer.state_dict()
        )
        checkpoint.save(
            is_best=False,
            filename=self.disc_save_path + f"/latest_model-{epoch}.model",
            best_model=self.disc_save_path + "/best_model.model"
        )

    def evaluate(self):
        self.gen.eval()
        total_stoi = 0.0
        total_ssnr = 0.0
        total_pesq = 0.0
        total_csig = 0.0
        total_cbak = 0.0
        total_covl = 0.0
        with torch.no_grad():
            count, total_eval_loss = 0, 0.0
            if self.device == 0:
                print(f"[+] epoch: {self.epoch}, validation start")
                # tqdm_obj = tqdm(total=len(self.validation_loader), leave=True)
            for i, (noisy, clean) in enumerate(self.validation_loader):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                c = torch.sqrt(noisy.size(-1) / torch.sum(noisy ** 2.0, dim=-1))
                noisy = torch.transpose(noisy, 0, 1)
                clean = torch.transpose(clean, 0, 1)
                noisy = torch.transpose(noisy * c, 0, 1)
                clean = torch.transpose(clean * c, 0, 1)
                del c
                noisy_spec = torch.stft(
                    noisy, self.frame_size, self.frame_shift,
                    window=torch.hamming_window(self.frame_size).to(self.device),
                    return_complex=False, onesided=True
                )
                clean_spec = torch.stft(
                    clean, self.frame_size, self.frame_shift,
                    window=torch.hamming_window(self.frame_size).to(self.device),
                    return_complex=False, onesided=True
                )
                noisy_spec = power_compress(noisy_spec).permute(0, 1, 3, 2)
                clean_spec = power_compress(clean_spec)
                clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
                clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)
                clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)

                est_real, est_imag = self.gen(noisy_spec)
                est_real = est_real.permute(0, 1, 3, 2)
                est_imag = est_imag.permute(0, 1, 3, 2)
                est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
                est_spec = power_uncompress(est_real, est_imag).squeeze(1)
                est_audio = torch.istft(
                    torch.complex(est_spec[..., 0], est_spec[..., 1]),
                    self.frame_size, self.frame_shift,
                    window=torch.hamming_window(self.frame_size).to(self.device),
                    return_complex=False, onesided=True
                )

                loss_complex = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
                loss_mag = F.mse_loss(est_mag, clean_mag)
                loss_time = torch.mean(torch.abs(est_audio - clean))
                loss = self.weight[0] * loss_complex + self.weight[1] * loss_mag + self.weight[2] * loss_time
                total_eval_loss += loss.data.item()
                del noisy_spec, clean_spec, clean_real, clean_imag, clean_mag, est_real, est_imag, est_spec, est_mag
                del loss_complex, loss_mag, loss_time
                torch.cuda.empty_cache()
                est_audio = est_audio.squeeze(0).cpu().numpy()
                clean = clean.squeeze(0).cpu().numpy()
                eval_metric = eval_composite(clean, est_audio, self.sr)
                total_pesq += eval_metric['pesq']
                total_ssnr += eval_metric['ssnr']
                total_stoi += eval_metric['stoi']
                total_cbak += eval_metric['cbak']
                total_csig += eval_metric['csig']
                total_covl += eval_metric['covl']
                count += 1
                if self.device == 0:
                    print(f"[+] epoch: {self.epoch}, iter: {i}, eval_loss: {loss:.6f}")
                    # tqdm_obj.update(1)
            total_eval_loss = torch.tensor(total_eval_loss).to(self.device)
            total_stoi = torch.tensor(total_stoi).to(self.device)
            total_pesq = torch.tensor(total_pesq).to(self.device)
            total_ssnr = torch.tensor(total_ssnr).to(self.device)
            total_cbak = torch.tensor(total_cbak).to(self.device)
            total_csig = torch.tensor(total_csig).to(self.device)
            total_covl = torch.tensor(total_covl).to(self.device)
            count = torch.tensor(count).to(self.device)
            distributed.all_reduce(total_eval_loss, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(total_stoi, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(total_pesq, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(total_ssnr, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(total_cbak, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(total_csig, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(total_covl, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(count, op=distributed.ReduceOp.SUM)
            if self.device == 0:
                avg_eval_loss = total_eval_loss / count
                avg_stoi = total_stoi / count
                avg_pesq = total_pesq / count
                avg_ssnr = total_ssnr / count
                avg_cbak = total_cbak / count
                avg_csig = total_csig / count
                avg_covl = total_covl / count
                val_info = f"[epoch: {self.epoch}]\n" \
                           f"\ttrain_loss: {avg_eval_loss:.6f}\n" \
                           f"\tlr: {self.lr:.6f}\n" \
                           f"\tSTOI: {avg_stoi:.4f}\n" \
                           f"\tCOVL: {avg_covl:.4f}\n" \
                           f"\tCBAK: {avg_cbak:.4f}\n" \
                           f"\tCSIG: {avg_csig:.4f}\n" \
                           f"\tSSNR: {avg_ssnr:.4f}\n" \
                           f"\tPESQ: {avg_pesq:.4f}\n"
                print('=' * 20 + "new result" + '=' * 20 + '\n' + val_info)
                self.now_val_loss = avg_eval_loss
                with open(self.val_results_file, 'a') as f:
                    f.write(val_info)
                if self.epoch % 10 == 0:
                    email(f"epoch_result:{self.epoch}\n", val_info)
                self.save_checkpoint(self.epoch)
            return

    def train(self):
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            self.gen.train()

            self.lr = 0.0004 * (0.98 ** ((epoch - 1) // 2))
            self.lr_disc = 0.0004 * (0.98 ** ((epoch - 1) // 2)) * 2
            for param_group, param_group_disc in zip(self.gen_optimizer.param_groups, self.disc_optimizer.param_groups):
                param_group['lr'] = self.lr
                param_group_disc['lr'] = self.lr_disc
                if self.device == 0:
                    print(f'[+] generator Learning rate adjusted to  {self.lr}')
                    print(f'[+] discriminator Learning rate adjusted to  {self.lr_disc}')
            if self.device == 0:
                print(f"[+] epoch: {epoch}, train start")
                # tqdm_obj = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch}", leave=True)
            for index, (noisy, clean) in enumerate(self.train_loader):
                one_labels = torch.ones(self.batch_size).to(self.device).float()
                torch.cuda.empty_cache()
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                c = torch.sqrt(noisy.size(-1) / torch.sum(noisy ** 2.0, dim=-1))
                noisy = torch.transpose(noisy, 0, 1)
                clean = torch.transpose(clean, 0, 1)
                noisy = torch.transpose(noisy * c, 0, 1)
                clean = torch.transpose(clean * c, 0, 1)
                del c

                noisy_spec = torch.stft(
                    noisy, self.frame_size, self.frame_shift,
                    window=torch.hamming_window(self.frame_size).to(self.device),
                    return_complex=False, onesided=True
                )
                clean_spec = torch.stft(
                    clean, self.frame_size, self.frame_shift,
                    window=torch.hamming_window(self.frame_size).to(self.device),
                    return_complex=False, onesided=True
                )
                noisy_spec = power_compress(noisy_spec)
                clean_spec = power_compress(clean_spec)
                noisy_real = noisy_spec[:, 0, :, :].unsqueeze(1)
                noisy_imag = noisy_spec[:, 1, :, :].unsqueeze(1)
                noisy_spec = noisy_spec.permute(0, 1, 3, 2)
                clean_real = clean_spec[:, 0, :, :].unsqueeze(1)
                clean_imag = clean_spec[:, 1, :, :].unsqueeze(1)

                self.gen_optimizer.zero_grad()
                est_real, est_imag = self.gen(noisy_spec)
                est_real = est_real.permute(0, 1, 3, 2)
                est_imag = est_imag.permute(0, 1, 3, 2)
                est_spec = power_uncompress(est_real, est_imag).squeeze(1)
                est_mag = torch.sqrt(est_real ** 2 + est_imag ** 2)
                clean_mag = torch.sqrt(clean_real ** 2 + clean_imag ** 2)
                noisy_mag = torch.sqrt(noisy_real ** 2 + noisy_imag ** 2)
                est_audio = torch.istft(
                    torch.complex(est_spec[..., 0], est_spec[..., 1]),
                    self.frame_size, self.frame_shift,
                    window=torch.hamming_window(self.frame_size).to(self.device),
                    return_complex=False, onesided=True
                )

                gen_metric = self.disc(est_mag, clean_mag)
                loss_gen = F.mse_loss(gen_metric.flatten(), one_labels)
                loss_complex = F.mse_loss(est_real, clean_real) + F.mse_loss(est_imag, clean_imag)
                loss_mag = F.mse_loss(est_mag, clean_mag)
                loss_time = torch.mean(torch.abs(est_audio - clean))
                train_loss = (self.weight[0] * loss_complex + self.weight[1] * loss_mag +
                              self.weight[2] * loss_time + self.weight[3] * loss_gen)
                del noisy_spec, noisy_real, noisy_imag, clean_spec, clean_real, clean_imag, est_real, est_imag, est_spec
                del loss_complex, loss_mag, loss_time, loss_gen
                torch.cuda.empty_cache()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=5, norm_type=2)
                self.gen_optimizer.step()
                train_loss = train_loss.item()
                torch.cuda.empty_cache()
                clean_audio = clean.cpu().numpy()
                noisy_audio = noisy.detach().cpu().numpy()
                est_audio = est_audio.detach().cpu().numpy()
                score_est = pesq_ssnr_score(clean_audio, est_audio)
                score_noisy = pesq_ssnr_score(clean_audio, noisy_audio)
                if score_est is not None and score_noisy is not None:
                    self.disc_optimizer.zero_grad()
                    enhance_metric = self.disc(est_mag.detach(), clean_mag)
                    max_metric = self.disc(clean_mag, clean_mag)
                    noisy_metric = self.disc(noisy_mag, clean_mag)
                    L_E = F.mse_loss(enhance_metric.flatten(), score_est)
                    L_N = F.mse_loss(noisy_metric.flatten(), score_noisy)
                    L_C = F.mse_loss(max_metric.flatten(), one_labels)
                    disc_loss = self_correcting_loss(self.disc, self.disc_optimizer, L_C, L_E, L_N)
                    del enhance_metric, noisy_metric, max_metric, noisy_audio, est_audio, clean, noisy
                    del est_mag, clean_mag, noisy_mag, score_est, score_noisy
                    torch.cuda.empty_cache()
                else:
                    print(f"[-] iter/epoch = {index + 1}/{epoch} some score is none")
                    disc_loss = torch.tensor([0.], requires_grad=True).to(self.device)
                disc_loss.backward()
                self.disc_optimizer.step()
                disc_loss = disc_loss.item()
                if self.device == 0:
                    print(f"[+] epoch: {epoch}, iter: {index}, train_loss: {train_loss:.6f}, disc_loss: {disc_loss:.6f}")
                    # tqdm_obj.update(1)
                    # tqdm_obj.set_postfix(
                    #     train_loss="%0.5f" % train_loss,
                    #     disc_loss="%0.5f" % disc_loss,
                    # )
            self.evaluate()


def ddp_setup(rank, world_size=2):
    """
    Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank, world_size):
    print(f"[+] rank: {rank}, world_size: {world_size}")
    random.seed(3407 + rank)
    torch.manual_seed(3407 + rank)
    global config
    config.prepare()
    config.batch_size = config.batch_size // world_size
    ddp_setup(rank, world_size)
    model = Net().to(rank)
    model_disc = Discriminator().to(rank)
    trainer = GAN_Trainer(model, model_disc, rank, is_load_epoch=None)
    try:
        trainer.train()
    except Exception as e:
        print(f"[-] rank: {rank} Error in training: {e}")
        distributed.destroy_process_group()
        return
    print(f"[+] rank: {rank} destory_process_group")
    distributed.destroy_process_group()
    print(f"[+] rank: {rank} finished training")


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
