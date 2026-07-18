"""Experiment A v2: isolated Wavelet U-Net local-frequency branch."""

import torch
import torch.nn.functional as F
from einops import rearrange

from model.DWTFreqNet import DWTFreqNet, Res_block


MODEL_BASE_COMMIT = "71dfeb348878517775af3df0767b54747f692c5d"


class DWTFreqNet_WULLE(DWTFreqNet):
    """DWTFreqNet with a four-encoder/three-decoder Wavelet U-Net LLE."""

    REMOVED_LOCAL_NODES = (
        "local_encoder1_2",
        "local_encoder2_2",
        "local_encoder3_2",
        "local_encoder1_3",
        "local_encoder2_3",
        "local_encoder1_4",
    )
    REMOVED_GLOBAL_PROJECTIONS = (
        "global_channel1_3",
        "global_channel2_3",
        "global_channel1_4",
    )

    def __init__(
        self,
        config,
        n_channels=1,
        n_classes=1,
        img_size=256,
        vis=False,
        mode="train",
        deepsuper=True,
        awgm_variant="awgm_original",
        awgm_allow_fallback=False,
    ):
        super().__init__(
            config=config,
            n_channels=n_channels,
            n_classes=n_classes,
            img_size=img_size,
            vis=vis,
            mode=mode,
            deepsuper=deepsuper,
            awgm_variant=awgm_variant,
            awgm_allow_fallback=awgm_allow_fallback,
        )
        for name in self.REMOVED_LOCAL_NODES + self.REMOVED_GLOBAL_PROJECTIONS:
            delattr(self, name)

        channels = config.base_channel
        self.wulle_decoder3 = self._make_layer(
            Res_block, channels * 16, channels * 8, 1
        )
        self.wulle_decoder2 = self._make_layer(
            Res_block, channels * 12, channels * 4, 1
        )
        self.wulle_decoder1 = self._make_layer(
            Res_block, channels * 6, channels * 2, 1
        )

        self.model_variant = "dwtfreqnet_wulle_a"
        self.model_base_commit = MODEL_BASE_COMMIT
        self.local_branch = "wavelet_unet"
        self.local_encoder_nodes = 4
        self.local_decoder_nodes = 3
        self.nested_local_nodes_removed = 6
        self.second_dwt = True
        self.awgm_position = "after_lle"
        self.ldrc_unchanged = True
        self.final_decoder_unchanged = True
        self.record_intermediate_features = False
        self.last_wulle_shapes = None
        self.last_wulle_statistics = None

    @property
    def experiment_metadata(self):
        return {
            "model_variant": self.model_variant,
            "model_base_commit": self.model_base_commit,
            "local_branch": self.local_branch,
            "local_encoder_nodes": self.local_encoder_nodes,
            "local_decoder_nodes": self.local_decoder_nodes,
            "nested_local_nodes_removed": self.nested_local_nodes_removed,
            "second_dwt": self.second_dwt,
            "awgm_position": self.awgm_position,
            "ldrc_unchanged": self.ldrc_unchanged,
            "final_decoder_unchanged": self.final_decoder_unchanged,
        }

    def _record_features(self, features, d1_hvd, d2_hvd):
        if self.training and not self.record_intermediate_features:
            return
        self.last_wulle_shapes = {
            name: tuple(tensor.shape) for name, tensor in features.items()
        }
        self.last_wulle_statistics = {
            "wulle_{}_norm".format(name): float(
                tensor.detach().float().square().mean().sqrt().cpu()
            )
            for name, tensor in features.items()
        }
        self.last_wulle_statistics.update({
            "wulle_D1_HVD_mean_abs": float(
                d1_hvd.detach().float().abs().mean().cpu()
            ),
            "wulle_D2_HVD_mean_abs": float(
                d2_hvd.detach().float().abs().mean().cpu()
            ),
        })

    def forward(self, x):
        x_input = self.inc(x)
        A1, LH1, HL1, HH1 = self.har(x_input)

        G1_input = self.conv_wavelet_inchannel_global(
            torch.cat([LH1, HL1, HH1], dim=1)
        )
        G1 = self.global_encoder1_1(G1_input)
        E1_input = self.conv_wavelet_inchannel_local(A1)
        E1 = self.local_encoder1_1(E1_input)

        A2, LH2, HL2, HH2 = self.har(E1)
        G2 = self.global_encoder2_1(torch.cat([LH2, HL2, HH2], dim=1))
        E2 = self.local_encoder2_1(A2)

        A3, LH3, HL3, HH3 = self.har(E2)
        G3 = self.global_encoder3_1(torch.cat([LH3, HL3, HH3], dim=1))
        E3 = self.local_encoder3_1(A3)

        A4, LH4, HL4, HH4 = self.har(E3)
        G4 = self.global_encoder4_1(torch.cat([LH4, HL4, HH4], dim=1))
        E4 = self.local_encoder4_1(A4)

        pred4_lh, pred4_hl, pred4_hh = torch.chunk(
            self.global_channel3_2(G4), 3, dim=1
        )
        U3 = self.inversehar(E4, pred4_lh, pred4_hl, pred4_hh)
        D3 = self.wulle_decoder3(torch.cat([E3, U3], dim=1))

        pred3_lh, pred3_hl, pred3_hh = torch.chunk(
            self.global_channel2_2(G3), 3, dim=1
        )
        U2 = self.inversehar(D3, pred3_lh, pred3_hl, pred3_hh)
        D2 = self.wulle_decoder2(torch.cat([E2, U2], dim=1))

        pred2_lh, pred2_hl, pred2_hh = torch.chunk(
            self.global_channel1_2(G2), 3, dim=1
        )
        U1 = self.inversehar(D2, pred2_lh, pred2_hl, pred2_hh)
        D1 = self.wulle_decoder1(torch.cat([E1, U1], dim=1))

        _, D1_LH, D1_HL, D1_HH = self.har(D1)
        D1_HVD = torch.cat([D1_LH, D1_HL, D1_HH], dim=1)
        _, D2_LH, D2_HL, D2_HH = self.har(D2)
        D2_HVD = torch.cat([D2_LH, D2_HL, D2_HH], dim=1)

        G1_2 = self.global_encoder1_2(torch.cat([G1, self.up(G2)], dim=1))
        G2_2 = self.global_encoder2_2(
            torch.cat([G2, D1_HVD, self.up(G3)], dim=1)
        )
        G3_2 = self.global_encoder3_2(
            torch.cat([G3, D2_HVD, self.up(G4)], dim=1)
        )
        G1_3 = self.global_encoder1_3(
            torch.cat([G1, G1_2, self.up(G2_2)], dim=1)
        )
        G2_3 = self.global_encoder2_3(
            torch.cat([G2, G2_2, D1_HVD, self.up(G3_2)], dim=1)
        )
        G1_4 = self.global_encoder1_4(
            torch.cat([G1, G1_2, G1_3, self.up(G2_3)], dim=1)
        )

        self._record_features(
            {
                "E1": E1,
                "E2": E2,
                "E3": E3,
                "E4": E4,
                "D1": D1,
                "D2": D2,
                "D3": D3,
                "G1": G1,
                "G2": G2,
                "G3": G3,
                "G4": G4,
                "G1_4": G1_4,
                "G2_3": G2_3,
                "G3_2": G3_2,
            },
            D1_HVD,
            D2_HVD,
        )

        f_input = x_input
        f1 = G1_4
        f2 = G2_3
        f3 = G3_2

        finput_A, finput_H, finput_V, finput_D = self.har(f_input)
        finput_AA, finput_HH, finput_VV, finput_DD = self.har(finput_A)
        finput_att = self.wave_att_input_t(
            finput_AA, finput_HH, finput_VV, finput_DD
        )
        finput_HHVVDD = self.stand_cahnnel_input(finput_att).flatten(2).permute(0, 2, 1)

        f1_A, f1_H, f1_V, f1_D = self.har(f1)
        f1_att = self.wave_att_f1(f1_A, f1_H, f1_V, f1_D)
        f1_HVD = self.stand_cahnnel1(f1_att).flatten(2).permute(0, 2, 1)

        f2_A, f2_H, f2_V, f2_D = self.har(f2)
        f2_att = self.wave_att_f2(f2_A, f2_H, f2_V, f2_D)
        f2_HVD = self.stand_cahnnel2(f2_att).flatten(2).permute(0, 2, 1)

        f3_A, f3_H, f3_V, f3_D = self.har(f3)
        f3_att = self.wave_att_f3(f3_A, f3_H, f3_V, f3_D)
        f3_HVD = self.stand_cahnnel3(f3_att).flatten(2).permute(0, 2, 1)

        _, _, hinput, winput = finput_att.shape
        _, _, h1, w1 = f1_att.shape
        _, _, h2, w2 = f2_att.shape
        _, _, h3, w3 = f3_att.shape

        f3_HVDe = self.TransTo3e(
            f3_HVD, torch.cat((finput_HHVVDD, f1_HVD, f2_HVD), dim=1)
        )
        f2_HVDe = self.TransTo2e(
            f2_HVD, torch.cat((finput_HHVVDD, f1_HVD, f3_HVDe), dim=1)
        )
        f1_HVDe = self.TransTo1e(
            f1_HVD, torch.cat((finput_HHVVDD, f2_HVDe, f3_HVDe), dim=1)
        )
        finput_HHVVDDe = self.TransTo_input(
            finput_HHVVDD, torch.cat((f1_HVDe, f2_HVDe, f3_HVDe), dim=1)
        )

        f3_HVDe = rearrange(f3_HVDe, "b (h w) c -> b c h w", h=h3, w=w3)
        f2_HVDe = rearrange(f2_HVDe, "b (h w) c -> b c h w", h=h2, w=w2)
        f1_HVDe = rearrange(f1_HVDe, "b (h w) c -> b c h w", h=h1, w=w1)
        finput_HHVVDDe = rearrange(
            finput_HHVVDDe,
            "b (h w) c -> b c h w",
            h=hinput,
            w=winput,
        )

        finput_HHVVDDe = self.wavel_channel_down_to_origin_x_inut(finput_HHVVDDe)
        finput_A = self.inversehar(finput_HHVVDDe, finput_HH, finput_VV, finput_DD)
        x_input_ldrc = self.inversehar(finput_A, finput_H, finput_V, finput_D)

        f1_HVDe = self.wavel_channel_down_to_origin_x1_global_output_1_4(f1_HVDe)
        G1_4_ldrc = self.inversehar(f1_HVDe, f1_H, f1_V, f1_D) + f1
        f2_HVDe = self.wavel_channel_down_to_origin_x2_global_output_2_3(f2_HVDe)
        G2_3_ldrc = self.inversehar(f2_HVDe, f2_H, f2_V, f2_D) + f2
        f3_HVDe = self.wavel_channel_down_to_origin_x3_global_output_3_2(f3_HVDe)
        G3_2_ldrc = self.inversehar(f3_HVDe, f3_H, f3_V, f3_D) + f3

        x4_global_output_de = self.decoder4_channel(G4)
        H4_de, V4_de, D4_de = torch.chunk(x4_global_output_de, 3, dim=1)
        x4_out = self.out4(E4 + H4_de + V4_de + D4_de)
        x3_local_input_de = self.inversehar(E4, H4_de, V4_de, D4_de)

        x3_global_output_de = self.decoder3_channel(G3_2_ldrc)
        H3_de, V3_de, D3_de = torch.chunk(x3_global_output_de, 3, dim=1)
        x3_local_output_de = self.decoder3_channel_local(D3 + x3_local_input_de)
        x3_out = self.out3(x3_local_output_de + H3_de + V3_de + D3_de)
        x2_local_input_de = self.inversehar(
            x3_local_output_de, H3_de, V3_de, D3_de
        )

        x2_global_output_de = self.decoder2_channel(G2_3_ldrc)
        H2_de, V2_de, D2_de = torch.chunk(x2_global_output_de, 3, dim=1)
        x2_local_output_de = self.decoder2_channel_local(D2 + x2_local_input_de)
        x2_out = self.out2(x2_local_output_de + H2_de + V2_de + D2_de)
        x1_local_input_de = self.inversehar(
            x2_local_output_de, H2_de, V2_de, D2_de
        )

        x1_global_output_de = self.decoder1_channel(G1_4_ldrc)
        H1_de, V1_de, D1_de = torch.chunk(x1_global_output_de, 3, dim=1)
        x1_local_output_de = self.decoder1_channel_local(D1 + x1_local_input_de)
        x1_out = self.out1(x1_local_output_de + H1_de + V1_de + D1_de)

        x1_local_final = self.inversehar(
            x1_local_output_de, H1_de, V1_de, D1_de
        )
        out = self.outc_global(
            self.from_input2out(x1_local_final + x_input_ldrc)
        )

        if not self.deepsuper:
            return torch.sigmoid(out)

        gt_5 = self.gt_conv5(x4_out)
        gt_4 = self.gt_conv4(x3_out)
        gt_3 = self.gt_conv3(x2_out)
        gt_2 = self.gt_conv2(x1_out)
        gt5 = F.interpolate(gt_5, scale_factor=16, mode="bilinear", align_corners=True)
        gt4 = F.interpolate(gt_4, scale_factor=8, mode="bilinear", align_corners=True)
        gt3 = F.interpolate(gt_3, scale_factor=4, mode="bilinear", align_corners=True)
        gt2 = F.interpolate(gt_2, scale_factor=2, mode="bilinear", align_corners=True)
        d0 = self.outconv(torch.cat((gt2, gt3, gt4, gt5, out), dim=1))

        if self.mode == "train":
            return (
                torch.sigmoid(gt5),
                torch.sigmoid(gt4),
                torch.sigmoid(gt3),
                torch.sigmoid(gt2),
                torch.sigmoid(d0),
                torch.sigmoid(out),
            )
        return torch.sigmoid(out)
