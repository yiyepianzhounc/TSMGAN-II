import torch
import torch.nn as nn
from utils import numParamsAll

class MaskFormerblock(nn.Module):
    def __init__(self, num_feature, num_head, bidirection=True, dropout=0.1):
        super(MaskFormerblock, self).__init__()
        self.muti_self_attention = nn.modules.activation.MultiheadAttention(num_feature, num_head, dropout=dropout)
        self.gru = nn.modules.rnn.GRU(num_feature, num_feature * 2, num_layers=1, bidirectional=bidirection)
        self.dropout = nn.modules.dropout.Dropout(dropout)
        if bidirection:
            self.linear = nn.modules.linear.Linear(num_feature * 4, num_feature)
        else:
            self.linear = nn.modules.linear.Linear(num_feature * 2, num_feature)
        self.norm1 = nn.modules.normalization.LayerNorm(num_feature)
        self.gelu = nn.GELU()

    def forward(self, x):
        x_mask = self.muti_self_attention(x, x, x, attn_mask=None, key_padding_mask=None)[0]
        self.gru.flatten_parameters()
        x_out, _ = self.gru(x)
        x_out = self.gelu(x_out)
        x_out = self.dropout(x_out)
        x_out = self.linear(x_out)
        x_out = torch.mul(x_out, x_mask)
        x_out = self.norm1(x_out)
        return x_out


class TwoStageTransformer(nn.Module):
    def __init__(self, input_csize, outpt_csize, dropout=0, num_layers=1):
        super(TwoStageTransformer, self).__init__()
        self.input_csize = input_csize  # 64
        self.output_csize = outpt_csize  # 64
        self.num_layers = num_layers  # 4
        self.input = nn.Sequential(
            nn.Conv2d(self.input_csize, self.input_csize // 2, kernel_size=1),
            nn.PReLU()
        )
        self.row_trans = nn.ModuleList([])
        self.col_trans = nn.ModuleList([])
        self.row_norm = nn.ModuleList([])
        self.col_norm = nn.ModuleList([])
        for i in range(self.num_layers):
            self.row_trans.append(
                MaskFormerblock(
                    self.input_csize // 2,
                    4,
                    dropout=dropout,
                    bidirection=True
                )
            )
            self.col_trans.append(
                MaskFormerblock(
                    self.input_csize // 2,
                    4,
                    dropout=dropout,
                    bidirection=True
                )
            )
            self.row_norm.append(nn.GroupNorm(1, self.input_csize // 2, eps=1e-8))
            self.col_norm.append(nn.GroupNorm(1, self.input_csize // 2, eps=1e-8))
        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(self.input_csize // 2, self.output_csize, kernel_size=1)
        )

    def forward(self, x):
        batch_size, channel_size, num_frames, frame_size = x.shape
        x = self.input(x)
        for i in range(self.num_layers):
            row = x.permute(3, 0, 2, 1).contiguous()
            # [frame_size, batch_size, num_frames, channel_size]
            row = row.view(frame_size, batch_size * num_frames, channel_size // 2)
            row = self.row_trans[i](row)
            row = row.view(frame_size, batch_size, num_frames, channel_size // 2)
            row = row.permute(1, 3, 2, 0).contiguous()
            # [batch_size, channel_size, frame_size, num_frames]
            row = self.row_norm[i](row)
            x = x + row

            col = x.permute(2, 0, 3, 1).contiguous()
            # [num_frames, batch_size, frame_size, channel_size]
            col = col.view(num_frames, batch_size * frame_size, channel_size // 2)
            col = self.col_trans[i](col)
            col = col.view(num_frames, batch_size, frame_size, channel_size // 2)
            col = col.permute(1, 3, 0, 2).contiguous()
            # [batch_size, channel_size, frame_size, num_frames]
            col = self.col_norm[i](col)
            x = x + col
        x = self.output(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, input_size, channel_size=64, depth=5):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.input_size = input_size
        self.channel_size = channel_size
        self.width = 2
        self.kernel_size = (self.width, 3)
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.width + (dil - 1) * (self.width - 1) - 1
            # 第四维增加2(左右各加一), 第三维增加(只在上加)pad_length
            setattr(self, f"pad{i + 1}", nn.ConstantPad2d((1, 1, pad_length, 0), value=0, ))
            # 第三,四维: 膨胀的卷积将pad后的x还原, 膨胀系数和pad_length一样与i变化
            # 第二维: 将size从 64 * i 转为 64
            setattr(
                self, f"conv{i + 1}",
                nn.Conv2d(
                    self.channel_size * (i + 1),
                    self.channel_size,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1)
                )
            )
            setattr(self, f"norm{i + 1}", nn.LayerNorm(self.input_size))
            setattr(self, f"prelu{i + 1}", nn.PReLU(self.channel_size))

    def forward(self, x):
        skip = x
        out = []
        for i in range(self.depth):
            x = getattr(self, f"pad{i + 1}")(skip)
            x = getattr(self, f"conv{i + 1}")(x)
            x = getattr(self, f"norm{i + 1}")(x)
            x = getattr(self, f"prelu{i + 1}")(x)
            out.append(x)
            skip = torch.cat([x, skip], dim=1)
        return out


class Encoder(nn.Module):
    def __init__(self, input_channel_size, output_channel_size, frame_size, denseblock_depth):
        super(Encoder, self).__init__()
        self.input_csize = input_channel_size  # 3
        self.outpt_csize = output_channel_size  # 64
        self.frame_size = frame_size  # 201
        self.denseblock_depth = denseblock_depth  # 4
        self.conv = nn.Conv2d(self.input_csize, self.outpt_csize, kernel_size=(1, 1))
        self.norm1 = nn.LayerNorm(self.frame_size)
        self.norm2 = nn.LayerNorm(self.frame_size)
        self.prelu1 = nn.PReLU(self.outpt_csize)
        self.prelu2 = nn.PReLU(self.outpt_csize)
        self.denseblock = DenseBlock(self.frame_size, self.outpt_csize, self.denseblock_depth)

    def forward(self, x, mag):
        out = torch.cat([mag, x], dim=1)
        out = self.conv(out)
        out = self.norm1(out)
        out = self.prelu1(out)
        x_out = self.denseblock(out)
        out = self.norm2(x_out[-1])
        out = self.prelu2(out)
        return out, x_out


class ConvAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2, kersize=[3, 5, 10], subband_num=1):
        super(ConvAttention, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.num_channels = num_channels
        self.smallConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[0], groups=num_channels // subband_num),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.middleConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[1], groups=num_channels // subband_num),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.largeConv1d = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, kernel_size=kersize[2], groups=num_channels // subband_num),
            nn.AdaptiveAvgPool1d(1),
            nn.ReLU(inplace=True)
        )
        self.feature_concate_fc = nn.Linear(3, 1, bias=True)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        output_tensor = input_tensor.permute(0, 1, 3, 2).contiguous()
        # [batch_size, channel_size, frame_size, num_frame]
        b, c, f, t = output_tensor.shape
        output_tensor = output_tensor.view(b, c * f, t)
        small_feature = self.smallConv1d(output_tensor)
        middle_feature = self.middleConv1d(output_tensor)
        large_feature = self.largeConv1d(output_tensor)
        feature = torch.cat([small_feature, middle_feature, large_feature], dim=2)
        feature = self.feature_concate_fc(feature)[..., 0]
        fc_out = self.relu(self.fc1(feature))
        fc_out = self.sigmoid(self.fc2(fc_out))
        t1, t2 = feature.size()
        output_tensor = torch.mul(output_tensor, fc_out.view(t1, t2, 1))
        output_tensor = output_tensor.reshape(b, c, f, t).permute(0, 1, 3, 2)
        output_tensor = output_tensor + input_tensor
        return output_tensor


class MultiAttention(nn.Module):
    def __init__(self, frame_size):
        super(MultiAttention, self).__init__()
        self.frame_size = frame_size
        self.channel_attention_mag = ConvAttention(self.frame_size)
        self.channel_attention_real = ConvAttention(self.frame_size)
        self.channel_attention_imag = ConvAttention(self.frame_size)

    def forward(self, x_mag, x_real, x_imag):
        x_mag = self.channel_attention_mag(x_mag)
        x_real = self.channel_attention_real(x_real)
        x_imag = self.channel_attention_imag(x_imag)
        x = torch.cat([x_real, x_imag], dim=1)
        return x_mag, x


class InfoCommunicate3(nn.Module):
    def __init__(self, channel_size):
        super(InfoCommunicate3, self).__init__()
        self.conv = nn.Conv2d(channel_size * 2, channel_size, kernel_size=1)
        self.output1 = nn.Sequential(
            nn.Conv2d(channel_size * 2, channel_size, kernel_size=1),
            nn.Tanh()
        )
        self.output2 = nn.Sequential(
            nn.Conv2d(channel_size * 2, channel_size, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feature_x1, feature_x2, feature_x3):
        out = torch.cat([feature_x2, feature_x3], dim=1)
        out = self.output1(out) * self.output2(out)
        out = torch.mul(feature_x1, out)
        return out


class Decoder(nn.Module):
    def __init__(self, input_channel_size, frame_size, denseblock_depth):
        super(Decoder, self).__init__()
        self.input_csize = input_channel_size  # 64
        self.frame_size = frame_size  # 201
        self.denseblock_depth = denseblock_depth  # 4
        self.width = 2
        self.kernel_size = (self.width, 3)
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.mag_conv = nn.Conv2d(self.input_csize, out_channels=self.input_csize // 2, kernel_size=(1, 1))
        self.mag_gru = nn.GRU(self.input_csize // 2, self.input_csize, num_layers=1, bidirectional=True)
        self.mag_linear = nn.Linear(self.input_csize * 2, self.input_csize // 2)
        self.mag_dec = nn.Sequential(
            nn.LayerNorm(self.frame_size),
            nn.PReLU(self.input_csize // 2),
            nn.Conv2d(self.input_csize // 2, out_channels=self.input_csize // 2, kernel_size=(1, 1)),
            nn.LayerNorm(self.frame_size),
            nn.PReLU(self.input_csize // 2),
            nn.Conv2d(self.input_csize // 2, out_channels=1, kernel_size=(1, 1)),
            nn.LayerNorm(self.frame_size),
            nn.PReLU(1)
        )
        self.complex_conv = nn.Conv2d(self.input_csize, out_channels=self.input_csize // 2, kernel_size=(1, 1))
        self.complex_gru = nn.GRU(self.input_csize // 2, self.input_csize, num_layers=1, bidirectional=True)
        self.complex_linear = nn.Linear(self.input_csize * 2, self.input_csize // 2)
        self.complex_dec = nn.Sequential(
            nn.LayerNorm(self.frame_size),
            nn.PReLU(self.input_csize // 2),
            nn.Conv2d(self.input_csize // 2, out_channels=self.input_csize // 2, kernel_size=(1, 1)),
            nn.LayerNorm(self.frame_size),
            nn.PReLU(self.input_csize // 2),
            nn.Conv2d(self.input_csize // 2, out_channels=2, kernel_size=(1, 1))
        )
        self.infoCom_c2m = InfoCommunicate3(input_channel_size)
        self.infoCom_m2c = InfoCommunicate3(input_channel_size)
        for i in range(self.denseblock_depth):
            dil = 2 ** i
            pad_length = self.width + (dil - 1) * (self.width - 1) - 1
            setattr(self, f"complex_pad{i + 1}", nn.ConstantPad2d((1, 1, pad_length, 0), value=0, ))
            setattr(
                self, f"complex_conv{i + 1}",
                nn.Conv2d(
                    self.input_csize * (i + 1),
                    self.input_csize,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1)
                )
            )
            setattr(self, f"complex_norm{i + 1}", nn.LayerNorm(self.frame_size))
            setattr(self, f"complex_prelu{i + 1}", nn.PReLU(self.input_csize))

            setattr(self, f"mag_pad{i + 1}", nn.ConstantPad2d((1, 1, pad_length, 0), value=0, ))
            setattr(
                self, f"mag_conv{i + 1}",
                nn.Conv2d(
                    self.input_csize * (i + 1),
                    self.input_csize,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1)
                )
            )
            setattr(self, f"mag_norm{i + 1}", nn.LayerNorm(self.frame_size))
            setattr(self, f"mag_prelu{i + 1}", nn.PReLU(self.input_csize))

    def forward(self, x, x_out):
        complex_out, complex_skip = x, x
        mag_out, mag_skip = x, x
        for i in range(self.denseblock_depth):
            complex_out = getattr(self, f"complex_pad{i + 1}")(complex_skip)
            complex_out = getattr(self, f"complex_conv{i + 1}")(complex_out)
            complex_out = getattr(self, f"complex_norm{i + 1}")(complex_out)
            complex_out = getattr(self, f"complex_prelu{i + 1}")(complex_out)

            mag_out = getattr(self, f"mag_pad{i + 1}")(mag_skip)
            mag_out = getattr(self, f"mag_conv{i + 1}")(mag_out)
            mag_out = getattr(self, f"mag_norm{i + 1}")(mag_out)
            mag_out = getattr(self, f"mag_prelu{i + 1}")(mag_out)

            complex_out = self.infoCom_m2c(complex_out, mag_out, x_out[i])
            mag_out = self.infoCom_c2m(mag_out, complex_out, x_out[i])
            complex_skip = torch.cat([complex_skip, complex_out], dim=1)
            mag_skip = torch.cat([mag_skip, mag_out], dim=1)

        mag_out = self.mag_conv(mag_out)
        b, c, t, f = mag_out.shape
        mag_out = mag_out.permute(3, 0, 2, 1).contiguous()
        mag_out = mag_out.view(self.frame_size, -1, self.input_csize // 2)
        mag_out = self.mag_gru(mag_out)[0]
        mag_out = self.mag_linear(mag_out)
        mag_out = mag_out.view(self.frame_size, -1, t, self.input_csize // 2)
        mag_out = mag_out.permute(1, 3, 2, 0).contiguous()
        mag_out = self.mag_dec(mag_out)

        complex_out = self.complex_conv(complex_out)
        complex_out = complex_out.permute(3, 0, 2, 1).contiguous()
        complex_out = complex_out.view(self.frame_size, -1, self.input_csize // 2)
        complex_out = self.complex_gru(complex_out)[0]
        complex_out = self.complex_linear(complex_out)
        complex_out = complex_out.view(self.frame_size, -1, t, self.input_csize // 2)
        complex_out = complex_out.permute(1, 3, 2, 0).contiguous()
        complex_out = self.complex_dec(complex_out)
        return mag_out, complex_out


class Net(nn.Module):
    def __init__(self, input_channel_size=3, width=64, frame_size=201):
        super(Net, self).__init__()
        self.input_csize = input_channel_size
        self.width = width  # 过渡的channel_size
        self.frame_size = frame_size
        self.ma_block = MultiAttention(self.frame_size)
        self.encoder = Encoder(self.input_csize, self.width, self.frame_size, denseblock_depth=4)
        self.ts_transformer = TwoStageTransformer(self.width, self.width, num_layers=4)
        self.decoder = Decoder(self.width, self.frame_size, denseblock_depth=4)

    def forward(self, x):
        x_real = x[:, 0, :, :].unsqueeze(1)
        x_imag = x[:, 1, :, :].unsqueeze(1)
        mag = torch.sqrt(x_real ** 2 + x_imag ** 2)
        phase = torch.angle(torch.complex(x_real, x_imag))
        mag, x = self.ma_block(mag, x_real, x_imag)
        del x_real, x_imag
        x, x_out = self.encoder(x, mag)
        x = self.ts_transformer(x)
        x_mag_mask, x_complex = self.decoder(x, x_out)
        x_mag = x_mag_mask * mag
        mag_real = x_mag * torch.cos(phase)
        mag_imag = x_mag * torch.sin(phase)
        mag_real = mag_real + x_complex[:, 0, :, :].unsqueeze(1)
        mag_imag = mag_imag + x_complex[:, 1, :, :].unsqueeze(1)
        return mag_real, mag_imag

if __name__ == '__main__':
    # x = torch.randn(4, 48, 3, 3).cuda()
    # model = DualTransformer(48, 48, 0, 4).cuda()
    # print(numParamsAll(model))
    # print(model(x).shape)
    x = torch.randn(4, 2, 321, 201).cuda()
    model = Net(width=64).cuda()
    with torch.no_grad():
        print(model(x)[0].shape)
    print(numParamsAll(model))
    print(numParamsAll(model.ma_block))
    print(numParamsAll(model.encoder))
    print(numParamsAll(model.ts_transformer))
    print(numParamsAll(model.decoder))
    # print(model(x).shape)
