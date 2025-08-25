import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
class DilatedBlock(nn.Module):
    """
    Single 3x3 dilated conv block (no padding -> spatial size shrinks).
    Conv -> (optional BN) -> GELU
    """
    def __init__(self, ch_in,ch_out, dilation, use_bn=False):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1,
                              padding=0, dilation=dilation, bias=not use_bn)
        self.bn = nn.BatchNorm2d(ch_out) if use_bn else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DilatedNet(nn.Module):

    def __init__(self, in_ch=3, out_ch=3, width=[8,16,32,32],
                 dilations=(1, 2, 4, 8), use_bn=False):
        super().__init__()
        # assert len(dilations) == 5, "Provide exactly 5 dilations."

        self.head = nn.Conv2d(in_ch, width[0], kernel_size=3, stride=1, padding=0, bias=True)

        blocks = []
        for idx,d in enumerate(dilations):
            if idx == len(dilations)-1:
              blocks.append(DilatedBlock(width[idx],width[idx], dilation=d, use_bn=use_bn))
            else:
              blocks.append(DilatedBlock(width[idx],width[idx+1], dilation=d, use_bn=use_bn))
        self.body = nn.Sequential(*blocks)

        self.tail = nn.Conv2d(width[-1], width[-1], kernel_size=3, stride=1, padding=0, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x_head = self.head(x)
        x = self.body(x_head)
        x = self.tail(x)
        return x,x_head


class SPPLayer(nn.Module):
    """
    Spatial Pyramid Pooling.
    Produces a fixed-length vector: C * sum(level^2) per sample, independent of input H,W.

    Args:
        levels: iterable of int, e.g., (1, 2, 3, 6) -> 1x1, 2x2, 3x3, 6x6 bins
        mode: 'max' or 'avg'
        flatten: if True, returns (N, C * sum(l*l)), else returns list of tensors per level
    """
    def __init__(self, levels=(8, 16, 32, 64), mode='avg', cat=True):
        super().__init__()
        assert mode in ('max', 'avg')
        self.levels = tuple(levels)
        self.mode = mode
        self.cat = cat

    def forward(self, x):
        # x: (N, C, H, W)
        _,_,H,W = x.size()
        assert x.dim() == 4, "SPP expects a 4D tensor (N,C,H,W)"
        pools = []
        for l in self.levels:
            if self.mode == 'max':
                p = F.adaptive_max_pool2d(x, output_size=(l, l))
            else:
                p = F.adaptive_avg_pool2d(x, output_size=(l, l))
                upsampled = F.interpolate(p, size=(H, W),mode="bilinear", align_corners=False)

            pools.append(upsampled)  # (N, C, l, l)

        
        if self.cat:
            return torch.cat(pools, dim=1)
        else:
            # Return a list of pooled feature maps per level
            return pools

    @property
    def output_multiplier(self):
        """sum of l*l; useful to compute output dim as C * output_multiplier"""
        return sum(l*l for l in self.levels)


class BoundaryRefineBlock(nn.Module):
    """
    Residual boundary refinement block.
    Two 3x3 convs (with dilation), BN, ReLU; preserves HxW via padding=dilation.
    """
    def __init__(self, ch, dilation=1, use_bn=True):
        super().__init__()
        bias = not use_bn
        pad = dilation

        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=pad, dilation=dilation, bias=bias)
        self.bn1   = nn.BatchNorm2d(ch) if use_bn else nn.Identity()
        self.act1  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=pad, dilation=dilation, bias=bias)
        self.bn2   = nn.BatchNorm2d(ch) if use_bn else nn.Identity()

        self.act_out = nn.ReLU(inplace=True)

        # He init
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        # Start near-identity: last conv gamma ~ 0 (optional but helpful)
        if use_bn:
            nn.init.zeros_(self.bn2.weight) if hasattr(self.bn2, "weight") and self.bn2.weight is not None else None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.act_out(out)



class FullyConnectedReshapeNet(nn.Module):
    def __init__(self,
                 in_shape=(32, 222, 222),
                 out_shape=(160, 254, 254),
                 hidden=1024):
        super().__init__()
        C_in, H_in, W_in = in_shape
        C_out, H_out, W_out = out_shape

        self.in_features = C_in * H_in * W_in
        self.out_features = C_out * H_out * W_out
        self.out_shape = out_shape

        self.fc = nn.Sequential(
            nn.Linear(self.in_features, hidden),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (N, C, H, W)
        N = x.size(0)
        x = x.view(N, -1)                  # flatten
        x = self.fc(x)                     # (N, out_features)
        x = x.view(N, *self.out_shape)     # reshape to (N, C_out, H_out, W_out)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.down = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.act(out + identity)
        return out


# ---- Two-branch fusion model ----
class TabImageRegressor(nn.Module):
    """
    Inputs:
      - X_tab: (N, 512)
      - X_img: (N, 168, 254, 254)
    Output:
      - y: (N, 1)
    """
    def __init__(self,
                 img_in_ch=168,
                 img_width=254, img_height=254,   # not strictly required, kept for clarity
                 img_feat_dim=512,                # image feature dim after pooling
                 tab_in_dim=512,                  # tabular/vector input size
                 hidden=512):                     # fusion MLP hidden size
        super().__init__()

        # ---- Image backbone (compact ResNet-ish) ----
        self.stem = nn.Sequential(
            nn.Conv2d(img_in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),  # 254->127
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                          # 127->64 (floor 63), but padding keeps ~64
        )
        self.layer1 = BasicBlock(64,   128, stride=2)  # ~64 -> ~32
        self.layer2 = BasicBlock(128,  256, stride=2)  # ~32 -> ~16
        self.layer3 = BasicBlock(256,  512, stride=2)  # ~16 -> ~8

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))   # -> (N, 512, 1, 1)
        self.img_proj = nn.Linear(512, img_feat_dim)   # -> (N, img_feat_dim)

        # ---- Tabular branch ----
        self.tab_norm = nn.LayerNorm(tab_in_dim)       # stable scaling for the 512-d vector
        self.tab_proj = nn.Identity()                  # keep as 512; replace with Linear if you want

        # ---- Fusion head ----
        fusion_in = img_feat_dim + tab_in_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1)                  # scalar output
        )

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=5**0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X_tab, X_img):
        # Image path
        z = self.stem(X_img)
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.avgpool(z).flatten(1)     # (N, 512)
        z = self.img_proj(z)               # (N, img_feat_dim)

        # Tabular path
        t = self.tab_norm(X_tab)           # (N, 512)
        t = self.tab_proj(t)               # (N, 512)

        # Fusion
        fused = torch.cat([z, t], dim=1)   # (N, img_feat_dim + 512)
        y = self.head(fused)               # (N, 1)
        return y

class ABSFusenet(nn.Module):
    def __init__(self):
        super().__init__()

        self.dn = DilatedNet()
        self.pooling = SPPLayer()
        self.br1 = BoundaryRefineBlock(168)
        
        self.res18 = torchvision.models.resnet18()
        class Identity(nn.Module):
            def __init__(self):
                super(Identity, self).__init__()
                
            def forward(self, x):
                return x

        self.res18.fc = Identity()
        self.regressor = TabImageRegressor()

    def forward(self,img):

        x,x_head = self.dn(img)
        B,C,H,W = x_head.size()
        x_pooled = self.pooling(x)
        x =  torch.cat([x_pooled,x],dim=1)
        x = F.interpolate(x, size=(H, W),mode="bilinear", align_corners=False)
        x = torch.cat([x_head,x],dim=1)

        y= self.res18(img)
        output = self.regressor(y,x)

        return output

if __name__ == "__main__":
    img = torch.rand((4,3,256,256))
    model = ABSFusenet()
    y = model(img)
    print(model.parameters())
    print("Output shape:", y.shape)  # (N, 1)


