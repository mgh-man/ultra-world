# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RTDETRDecoder", "v10Detect"


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=365, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1).contiguous(), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1).contiguous() for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2).contiguous(), cls.sigmoid().permute(0, 2, 1).contiguous()
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """
    WorldDetect头部模块，用于集成YOLO检测模型与文本嵌入的语义理解。
    
    这是一个扩展的检测头，支持基于文本嵌入的对比学习，
    可以将视觉特征与文本特征进行对齐，实现更好的语义理解。
    """

    def __init__(self, nc=365, embed=512, with_bn=False, ch=()):
        """
        初始化WorldDetect检测层。
        
        Args:
            nc (int): 类别数量，默认80
            embed (int): 嵌入维度，默认512
            with_bn (bool): 是否使用BatchNorm的对比头，默认False
            ch (tuple): 输入通道数元组
        """
        super().__init__(nc, ch)
        # 计算类别头的通道数，取输入通道的最大值和类别数的最小值(不超过100)
        c3 = max(ch[0], min(self.nc, 100))
        # 创建用于生成嵌入特征的卷积层序列
        # 每层包含: Conv(3x3) -> Conv(3x3) -> Conv2d(1x1输出embed维度)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        # 创建对比学习头，根据with_bn选择是否使用BatchNorm版本
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """
        前向传播，计算边界框和类别概率。
        
        Args:
            x (list): 多尺度特征图列表
            text: 文本嵌入特征
            
        Returns:
            训练时返回原始预测，推理时返回处理后的预测结果
        """
        # 对每个检测层进行处理
        for i in range(self.nl):
            # 拼接回归分支(cv2)和对比学习分支(cv4+cv3)的输出
            # cv2: 边界框回归特征
            # cv4(cv3): 对比学习头处理嵌入特征与文本特征
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        
        # 训练模式直接返回原始特征
        if self.training:
            return x

        # 推理路径 - 后处理预测结果
        shape = x[0].shape  # 获取特征图形状 BCHW
        # 将多尺度特征重组为统一格式: [batch, channels, total_anchors]
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        
        # 动态生成锚点和步长(如果形状改变)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1).contiguous() for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        # 根据导出格式分离边界框和类别预测
        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # 避免TF FlexSplitV操作
            box = x_cat[:, : self.reg_max * 4]  # 边界框预测
            cls = x_cat[:, self.reg_max * 4 :]  # 类别预测
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        # 针对TFLite/EdgeTPU格式的特殊处理
        if self.export and self.format in {"tflite", "edgetpu"}:
            # 预计算归一化因子以提高数值稳定性
            # 参考: https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            # 标准边界框解码: DFL解码 + 锚点偏移 + 步长缩放
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        # 拼接解码后的边界框和Sigmoid激活的类别概率
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """
        初始化检测层的偏置项。
        警告: 需要步长(stride)已经可用。
        """
        m = self  # 获取当前Detect模块
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # 标称类别频率
        
        # 遍历回归分支(cv2)、嵌入分支(cv3)和对应的步长
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0  # 初始化边界框回归的偏置为1.0
            # 注释掉类别预测的偏置初始化，因为WorldDetect使用对比学习而非传统分类
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    实时可变形Transformer解码器（RTDETRDecoder）模块，用于目标检测。

    该解码器模块利用Transformer架构结合可变形卷积来预测边界框和类别标签。
    它整合来自多个层的特征，通过一系列Transformer解码器层运行以输出最终预测。
    """

    export = False  # 导出模式标志

    def __init__(
        self,
        nc=80,  # 类别数量
        ch=(512, 1024, 2048),  # 主干网络特征图通道数
        hd=256,  # 隐藏层维度
        nq=300,  # 查询数量
        ndp=4,  # 解码器点数量
        nh=8,  # 多头注意力头数
        ndl=6,  # 解码器层数
        d_ffn=1024,  # 前馈网络维度
        dropout=0.0,  # Dropout率
        act=nn.ReLU(),  # 激活函数
        eval_idx=-1,  # 评估索引
        # 训练参数
        nd=100,  # 去噪数量
        label_noise_ratio=0.5,  # 标签噪声比例
        box_noise_scale=1.0,  # 边界框噪声缩放
        learnt_init_query=False,  # 是否学习初始查询嵌入
    ):
        """
        使用给定参数初始化RTDETRDecoder模块。

        Args:
            nc (int): 类别数量，默认80
            ch (tuple): 主干网络特征图的通道数，默认(512, 1024, 2048)
            hd (int): 隐藏层维度，默认256
            nq (int): 查询点数量，默认300
            ndp (int): 解码器点数量，默认4
            nh (int): 多头注意力的头数，默认8
            ndl (int): 解码器层数，默认6
            d_ffn (int): 前馈网络维度，默认1024
            dropout (float): Dropout率，默认0
            act (nn.Module): 激活函数，默认nn.ReLU
            eval_idx (int): 评估索引，默认-1
            nd (int): 去噪数量，默认100
            label_noise_ratio (float): 标签噪声比例，默认0.5
            box_noise_scale (float): 边界框噪声缩放，默认1.0
            learnt_init_query (bool): 是否学习初始查询嵌入，默认False
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # 特征层级数量
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # 主干网络特征投影层
        # 将不同通道数的特征图投影到统一的隐藏维度
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # 注意: 简化版本但与.pt权重不一致
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer模块
        # 创建可变形Transformer解码器层
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        # 创建完整的解码器，包含多个解码器层
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # 去噪部分 - 用于训练时的噪声注入
        self.denoising_class_embed = nn.Embedding(nc, hd)  # 类别去噪嵌入
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # 解码器嵌入
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            # 如果启用，学习初始查询嵌入
            self.tgt_embed = nn.Embedding(nq, hd)
        # 查询位置编码头
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # 编码器输出头
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))  # 编码器特征处理
        self.enc_score_head = nn.Linear(hd, nc)  # 编码器分类头
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)  # 编码器边界框回归头

        # 解码器输出头
        # 为每个解码器层创建独立的分类和回归头
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        # 初始化参数
        self._reset_parameters()

    def forward(self, x, batch=None):
        """执行模块的前向传播，返回输入的边界框和分类分数。"""
        from ultralytics.models.utils.ops import get_cdn_group

        # 输入投影和嵌入
        feats, shapes = self._get_encoder_input(x)

        # 准备去噪训练数据
        # 获取去噪组，用于训练时的一致性正则化
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        # 获取解码器输入
        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # 解码器前向传播
        # 通过Transformer解码器处理查询和特征，输出最终预测
        dec_bboxes, dec_scores = self.decoder(
            embed,  # 查询嵌入
            refer_bbox,  # 参考边界框
            feats,  # 编码器特征
            shapes,  # 特征图形状
            self.dec_bbox_head,  # 解码器边界框头
            self.dec_score_head,  # 解码器分类头
            self.query_pos_head,  # 查询位置头
            attn_mask=attn_mask,  # 注意力掩码
        )
        
        # 整合所有输出
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
            
        # 推理模式：合并边界框和分类分数 (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """为给定形状生成锚点边界框，使用特定网格大小并验证它们。"""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            # 创建网格坐标
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            # 生成网格
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            # 归一化坐标到[0,1]范围
            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            
            # 为不同层级设置不同的锚点尺寸
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        # 合并所有层级的锚点
        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        
        # 创建有效掩码，过滤边界附近的锚点
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        
        # 将锚点坐标转换到logit空间
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """处理并返回编码器输入，通过获取输入的投影特征并连接它们。"""
        # 获取投影特征 - 将不同通道的特征投影到统一维度
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        
        # 准备编码器输入
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c] 展平空间维度
            feats.append(feat.flatten(2).permute(0, 2, 1).contiguous())
            # 记录特征图形状 [nl, 2]
            shapes.append([h, w])

        # 连接所有层级的特征 [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """从提供的特征和形状生成并准备解码器所需的输入。"""
        bs = feats.shape[0]
        
        # 为解码器准备输入
        # 生成锚点和有效掩码
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        
        # 处理编码器特征，只保留有效位置的特征
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        # 编码器预测分数
        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # 查询选择 - 选择最有信心的位置作为查询
        # (bs, num_queries) 获取每个位置最大类别分数，选择topk
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries) 批次索引
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # 提取top-k特征和对应的锚点
        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # 动态锚点 + 静态内容
        # 通过边界框头预测锚点偏移，与原始锚点相加得到参考边界框
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        # 编码器边界框预测（sigmoid激活到[0,1]范围）
        enc_bboxes = refer_bbox.sigmoid()
        
        # 如果有去噪边界框，将其与参考边界框连接
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
            
        # 编码器分类分数
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        # 准备解码器嵌入
        # 根据配置选择学习的查询嵌入或使用top-k特征
        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        
        # 训练时分离梯度以稳定训练
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
                
        # 如果有去噪嵌入，将其与查询嵌入连接
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """使用预定义的权重和偏置初始化或重置模型各组件的参数。"""
        # 类别和边界框头初始化
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        
        # 注意: `linear_init`中的权重初始化在使用自定义数据集训练时会导致NaN
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)  # 设置分类偏置
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)  # 边界框头最后一层权重置零
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)    # 边界框头最后一层偏置置零
        
        # 初始化所有解码器层的分类和回归头
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)  # 分类偏置
            constant_(reg_.layers[-1].weight, 0.0)  # 回归权重置零
            constant_(reg_.layers[-1].bias, 0.0)    # 回归偏置置零

        # 初始化编码器输出层
        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        
        # 如果使用学习的初始查询，初始化查询嵌入
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
            
        # 初始化查询位置头
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        
        # 初始化输入投影层
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
