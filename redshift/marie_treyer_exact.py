from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from analysis_utils import compute_marie_regular_cv_indices, z_to_bin_indices
from config import CONFIG as LOCAL_CONFIG


MARIE_CONFIG: Dict[str, object] = {
    "BATCH_NORM": False,
    "ONE_CONV_AFTER_INCEPTIONS": False,
    "MODALITY_DROP_OUT": None,
    "SPECEFIC_MODALITIES_SWITCH_OFF": None,
    "MOD_DO_PROBA": 0.0,
    "VAL_MODALITIES_TO_SWITCH_OFF": None,
    "USE_CROSS_FUSION": False,
    "USE_MODALITY_TRANSFORMERS": False,
    "USE_CNN_ADV": True,
    "CNN_INPUT_STAGE": 1,
}


def marie_z_edges(n_bins: int = 360, z_min: float = 0.0, z_max: float = 6.0) -> np.ndarray:
    return np.linspace(float(z_min), float(z_max), int(n_bins) + 1, dtype=np.float64)


def marie_z_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def marie_point_estimate(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    pdf = torch.softmax(logits, dim=1)
    return torch.sum(pdf * bin_centers.view(1, -1), dim=1)


def compute_padding_same(in_dim: float, k_size: int, stride: int) -> int:
    return int(np.ceil(((stride - 1) * in_dim - stride + k_size) / 2))


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_dim: float,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
        acti_func: str = "relu",
        apply_pad: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        padding = "same" if apply_pad else "valid"
        self.conv = nn.Conv2d(in_channels, out_channels, bias=bias, kernel_size=kernel_size, padding=padding, stride=stride, **kwargs)
        if MARIE_CONFIG["BATCH_NORM"]:
            self.BN = nn.BatchNorm2d(out_channels)
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            self.conv.bias.data.fill_(0.0)

        if acti_func == "relu":
            self.acti_func = nn.ReLU()
        elif acti_func == "prelu":
            self.acti_func = nn.PReLU()
        elif acti_func == "tanh":
            self.acti_func = nn.Tanh()
        elif acti_func == "iden":
            self.acti_func = identity
        elif acti_func == "swish":
            self.acti_func = nn.SiLU()
        else:
            raise ValueError(f"Activation inconnue: {acti_func}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if MARIE_CONFIG["BATCH_NORM"]:
            x = self.BN(x)
        return self.acti_func(x)


class BasicFC(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        activation: bool = True,
        acti_func: str = "relu",
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(int(in_dim), out_dim, bias=bias)
        nn.init.xavier_uniform_(self.fc.weight)
        if bias:
            self.fc.bias.data.fill_(0.1)
        self.activation = activation

        if acti_func == "relu":
            self.acti_func = nn.ReLU()
        elif acti_func == "prelu":
            self.acti_func = nn.PReLU()
        elif acti_func == "sigmoid":
            self.acti_func = nn.Sigmoid()
        elif acti_func == "tanh":
            self.acti_func = nn.Tanh()
        elif acti_func == "iden":
            self.acti_func = identity
        elif acti_func == "swish":
            self.acti_func = nn.SiLU()
        else:
            raise ValueError(f"Activation inconnue: {acti_func}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return self.acti_func(x) if self.activation else x


def before_inception_block(
    archi: Sequence[int],
    n_channels_in: int,
    activations: Sequence[str],
    in_dim: Sequence[float] = (64, 64),
    kernel_size: int = 5,
    groups: int = 1,
) -> nn.Sequential:
    layers = [
        BasicConv2d(
            n_channels_in if i == 0 else int(archi[i - 1]),
            int(archi[i]),
            kernel_size=kernel_size,
            in_dim=in_dim[0],
            acti_func=activations[i],
            groups=groups,
        )
        for i in range(len(archi))
    ]
    return nn.Sequential(*layers)


def simple_inception_block(
    archi: int,
    in_channels: int,
    in_dims: Sequence[float],
    start_with_pooling: bool,
    add_dropout: bool = False,
    do_rate: float = 0.0,
    groups: int = 1,
) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    if start_with_pooling:
        layers.append(nn.AvgPool2d(2, stride=2))
    layers.append(BasicConv2d(in_channels, int(archi), kernel_size=3, in_dim=in_dims[0], acti_func="relu", groups=groups))
    if add_dropout:
        layers.append(nn.Dropout(p=do_rate))
    return nn.Sequential(*layers), int(archi)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 96,
        hidden_dims: Sequence[int] = (64, 128, 256),
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        layer_dims = [input_dim] + list(hidden_dims) + [output_dim]
        layers: List[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layer = nn.Linear(in_dim, out_dim, bias=use_bias)
            init.xavier_uniform_(layer.weight, gain=init.calculate_gain("relu"))
            if use_bias:
                init.zeros_(layer.bias)
            layers.append(layer)
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Parellel_Inception(nn.Module):
    def __init__(
        self,
        in_dim: Sequence[int],
        modalities: Sequence[Sequence[int]],
        before_inception_archi: Sequence[int] = (32, 32),
        inception_archi: Sequence[int] = (42,),
        pooling_before_inceptions: Sequence[bool] = (True, False),
    ) -> None:
        super().__init__()
        self.modalities = [list(m) for m in modalities]
        self.parallel_blocks = []
        self.out_n_channels = 0

        before_archi_by_mod = [[int(e * len(self.modalities[i])) for e in before_inception_archi] for i in range(len(self.modalities))]
        inception_in_dims = [[e / 2 ** sum(pooling_before_inceptions[0 : i + 1]) for e in in_dim] for i in range(len(inception_archi))]
        self.outs_dims = inception_in_dims[-1] if len(inception_in_dims) > 0 else list(in_dim)
        inception_archi_by_mod = [(np.asarray(inception_archi) * len(self.modalities[i])).astype(int).tolist() for i in range(len(self.modalities))]

        activations = ["prelu", "tanh"]
        if MARIE_CONFIG["ONE_CONV_AFTER_INCEPTIONS"]:
            activations = ["tanh"]

        for i in range(len(self.modalities)):
            before = before_inception_block(
                before_archi_by_mod[i],
                n_channels_in=len(self.modalities[i]),
                activations=activations,
            )
            in_c = before_archi_by_mod[i][-1]
            block_dims = [[e / 2 ** sum(pooling_before_inceptions[0 : j + 1]) for e in in_dim] for j in range(len(inception_archi_by_mod[i]))]
            blocks: List[nn.Module] = [before]
            for j in range(len(inception_archi_by_mod[i])):
                block, in_c = simple_inception_block(inception_archi_by_mod[i][j], in_c, block_dims[j], pooling_before_inceptions[j])
                blocks.append(block)
            self.out_n_channels += in_c
            self.parallel_blocks.append(nn.Sequential(*blocks))

        self.parallel_blocks = nn.ModuleList(self.parallel_blocks)
        if MARIE_CONFIG["USE_CROSS_FUSION"]:
            self.fusion_conv = BasicConv2d(self.out_n_channels, int(self.out_n_channels / len(self.modalities) + 1), self.outs_dims, kernel_size=3, apply_pad=True)
            self.out_n_channels = int(self.out_n_channels / len(self.modalities) + 1) * (len(self.modalities) + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if MARIE_CONFIG["USE_CROSS_FUSION"]:
            return self.forward_cross_fusion(x)
        return self.forward_concat(x)

    def forward_concat(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        if MARIE_CONFIG["USE_MODALITY_TRANSFORMERS"] and x.dim() == 5:
            for i in range(len(self.parallel_blocks)):
                outputs.append(self.parallel_blocks[i](x[:, i, :, :, :]))
        else:
            for i in range(len(self.parallel_blocks)):
                outputs.append(self.parallel_blocks[i](x[:, self.modalities[i], :, :]))
        return torch.cat(outputs, dim=1)

    def forward_cross_fusion(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [[None for _ in range(len(self.modalities))] for _ in range(len(self.modalities))]
        parallel_processed = []
        for i in range(len(self.modalities)):
            parallel_processed.append(self.parallel_blocks[i][0:2](x[:, self.modalities[i], :, :]))
        for i in range(len(self.modalities)):
            for j in range(len(self.modalities)):
                outputs[i][j] = self.parallel_blocks[i][2](parallel_processed[j])
        fused = [None for _ in range(len(self.modalities) + 1)]
        fused[len(self.modalities)] = []
        for i in range(len(self.modalities)):
            fused[i] = self.fusion_conv(torch.cat(outputs[:][i], dim=1))
            fused[len(self.modalities)].append(sum(outputs[i][:]))
        fused[len(self.modalities)] = self.fusion_conv(torch.cat(fused[len(self.modalities)], dim=1))
        return torch.cat(fused, dim=1)


class Model_multi_modal_simple(nn.Module):
    def __init__(
        self,
        in_dim: Sequence[int],
        n_outputs: int,
        modalities: Sequence[Sequence[int]],
        mags_input_size: Optional[int],
        parallel_before_inception_archi: Optional[Sequence[int]] = None,
        parallel_inception_archi: Optional[Sequence[int]] = None,
        inception_archi: Optional[Sequence[int]] = None,
        parallel_pooling_before_inceptions: Optional[Sequence[bool]] = None,
        pooling_before_inceptions: Optional[Sequence[bool]] = None,
        convs_after_inception: Sequence[int] = (96, 96, 96),
        convs_after_inception_pad: Sequence[bool] = (False, False, False),
        first_FC_dim: int = 1024,
        classification_FCs_archi_: Optional[List[int]] = None,
        regression_FCs_archi_: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if parallel_before_inception_archi is None:
            parallel_before_inception_archi = [32, 32]
        if parallel_inception_archi is None:
            parallel_inception_archi = [42]
        if inception_archi is None:
            inception_archi = [156, 156, 128, 128]
        if parallel_pooling_before_inceptions is None:
            parallel_pooling_before_inceptions = [True]
        if pooling_before_inceptions is None:
            pooling_before_inceptions = [False, True, False, True]
        if classification_FCs_archi_ is None:
            classification_FCs_archi_ = [1024]
        if regression_FCs_archi_ is None:
            regression_FCs_archi_ = [512]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_dim = list(in_dim)
        self.n_outputs = n_outputs
        self.avg_pool = nn.AvgPool2d(2, stride=2)
        self.modalities = [list(m) for m in modalities]
        self.parrellel_block = Parellel_Inception(
            in_dim,
            modalities=self.modalities,
            before_inception_archi=parallel_before_inception_archi,
            inception_archi=parallel_inception_archi,
            pooling_before_inceptions=parallel_pooling_before_inceptions,
        )

        inception_blocks = []
        in_c = self.parrellel_block.out_n_channels
        inception_in_dims = [[e / 2 ** sum(pooling_before_inceptions[0 : i + 1]) for e in self.parrellel_block.outs_dims] for i in range(len(inception_archi))]
        for i in range(len(inception_archi)):
            block, in_c = simple_inception_block(inception_archi[i], in_c, inception_in_dims[i], pooling_before_inceptions[i])
            inception_blocks.append(block)
        self.inception_blocks = nn.Sequential(*inception_blocks)
        out_c = in_c

        feature_map_size_output = [e / 2 ** sum(pooling_before_inceptions) for e in self.parrellel_block.outs_dims]
        self.convs_after_inceptions_output = [feature_map_size_output[0] - (2 * (i + 1)) for i in range(len(convs_after_inception))]

        if MARIE_CONFIG["ONE_CONV_AFTER_INCEPTIONS"]:
            self.convs_after_inceptions = nn.Sequential(OrderedDict([
                ("avgpool_4x4_stride_4", nn.AvgPool2d((4, 4), stride=(4, 4))),
                ("conv3x3_0", BasicConv2d(out_c, convs_after_inception[0], kernel_size=3, in_dim=feature_map_size_output[0], acti_func="relu", apply_pad=True)),
                ("avgpool_2x2_stride_2", nn.AvgPool2d((2, 2), stride=(2, 2))),
            ]))
        else:
            self.convs_after_inceptions = nn.Sequential(OrderedDict([
                ("conv3x3_0", BasicConv2d(out_c, convs_after_inception[0], kernel_size=3, in_dim=feature_map_size_output[0], acti_func="relu", apply_pad=convs_after_inception_pad[0])),
                ("conv3x3_1", BasicConv2d(convs_after_inception[0], convs_after_inception[1], kernel_size=3, in_dim=self.convs_after_inceptions_output[0], acti_func="relu", apply_pad=convs_after_inception_pad[1])),
                ("conv3x3_2", BasicConv2d(convs_after_inception[1], convs_after_inception[2], kernel_size=3, in_dim=self.convs_after_inceptions_output[1], acti_func="relu", apply_pad=convs_after_inception_pad[2])),
                ("avgpool_2x2_stride_1", nn.AvgPool2d((2, 2), stride=(2, 2))),
            ]))

        feature_after_conv = [self.convs_after_inceptions_output[2] / 2, self.convs_after_inceptions_output[2] / 2]
        fc_in = feature_after_conv[0] * feature_after_conv[1] * convs_after_inception[-1]
        fc_in += 1
        if mags_input_size is not None:
            output_size = 96
            self.mags_FC = MLP(input_dim=mags_input_size, output_dim=output_size, hidden_dims=[64, 128, 256], use_bias=True)
            fc_in += output_size

        self.first_FC = BasicFC(int(fc_in), first_FC_dim, activation=True)

        classification_FCs = []
        fc_current = first_FC_dim
        for i, out_dim in enumerate(list(classification_FCs_archi_) + [self.n_outputs]):
            classification_FCs.append(BasicFC(fc_current, out_dim, activation=i < len(classification_FCs_archi_)))
            fc_current = out_dim
        self.classification_FCs = nn.Sequential(*classification_FCs)

        regression_FCs = []
        fc_current = first_FC_dim
        for i, out_dim in enumerate(list(regression_FCs_archi_) + [1]):
            regression_FCs.append(BasicFC(fc_current, out_dim, activation=i < len(regression_FCs_archi_)))
            fc_current = out_dim
        self.regression_FCs = nn.Sequential(*regression_FCs)

        if MARIE_CONFIG["MODALITY_DROP_OUT"] is not None:
            self.total_number_modalities_bands = len([band for modality in self.modalities for band in modality])

    def modality_DO(self, x: torch.Tensor) -> torch.Tensor:
        modality_factors = torch.ones(1, self.total_number_modalities_bands, int(x.shape[1] / self.total_number_modalities_bands), device=x.device)
        factors = torch.ones(1, self.total_number_modalities_bands, 1, device=x.device)
        if self.training:
            modalities_to_switch_off = []
        else:
            modalities_to_switch_off = MARIE_CONFIG["VAL_MODALITIES_TO_SWITCH_OFF"]
        if modalities_to_switch_off is not None and len(modalities_to_switch_off) > 0:
            switched = []
            for modality in modalities_to_switch_off:
                start = sum(len(self.modalities[i]) for i in range(0, modality))
                end = start + len(self.modalities[modality])
                switched += list(range(start, end))
            factors *= self.total_number_modalities_bands / (self.total_number_modalities_bands - len(switched))
            factors[0, switched, 0] *= 0
        return x * (modality_factors * factors).flatten(start_dim=1)[:, :, None, None]

    def forward(
        self,
        X: torch.Tensor,
        ebv: Optional[torch.Tensor] = None,
        return_latent_repr: bool = False,
        mags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.parrellel_block(X)
        if MARIE_CONFIG["MODALITY_DROP_OUT"] is not None:
            X = self.modality_DO(X)
        latent_repr_cnn = None
        for i in range(len(self.inception_blocks)):
            X = self.inception_blocks[i](X)
            if i == MARIE_CONFIG["CNN_INPUT_STAGE"]:
                latent_repr_cnn = X
        X = self.convs_after_inceptions(X)

        latent_repr = torch.flatten(X, start_dim=1)
        if mags is not None:
            latent_repr = torch.cat((latent_repr, self.mags_FC(mags)), dim=1)
        if ebv is None:
            ebv = torch.zeros(latent_repr.shape[0], dtype=latent_repr.dtype, device=latent_repr.device)
        X = torch.cat((latent_repr, ebv.view(-1, 1)), dim=1)
        X = self.first_FC(X)
        if return_latent_repr:
            return self.classification_FCs(X), self.regression_FCs(X), latent_repr_cnn if MARIE_CONFIG["USE_CNN_ADV"] else latent_repr
        return self.classification_FCs(X), self.regression_FCs(X)


def build_marie_treyer_model(n_bins: int = 360, mags_input_size: int = 6) -> Model_multi_modal_simple:
    return Model_multi_modal_simple(
        in_dim=[LOCAL_CONFIG.IMG_SIZE, LOCAL_CONFIG.IMG_SIZE],
        n_outputs=n_bins,
        modalities=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
        mags_input_size=mags_input_size,
        parallel_before_inception_archi=[32, 32],
        parallel_inception_archi=[42],
        inception_archi=[156, 156, 128, 128],
        parallel_pooling_before_inceptions=[True],
        pooling_before_inceptions=[False, True, False, True],
    )


def marie_fold_split(n_samples: int, fold_id: int, n_folds: int = 5) -> Dict[str, np.ndarray]:
    return compute_marie_regular_cv_indices(n_samples, n_folds=n_folds, fold_id=fold_id, seed=42)


def z_to_marie_bins(z_values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return z_to_bin_indices(z_values, edges)
