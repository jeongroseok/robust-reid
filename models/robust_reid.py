import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.retrieval.average_precision import RetrievalMAP

from .components import *


class RobustReID(pl.LightningModule, Encoder, Generator):
    class __HPARAMS:
        backbone_type: str
        related_latent_dim: int
        unrelated_latent_dim: int
        num_classes: int
        learning_rate: float
        related_coeff: float
        unrelated_coeff: float
        img_recon_coeff: float
        code_recon_coeff: float
        adv_coeff: float
        id_coeff: float
        finetuning_backbone: bool

    hparams: __HPARAMS
    backbone: ResnetBackbone
    related_encoder: RelatedEncoder
    unrelated_encoder: UnrelatedEncoder
    classifier: BasicClassifer
    generator: ISGAN_Generator
    discriminator: Discriminator

    def __init__(
        self,
        backbone_type: str = "resnet18",
        related_latent_dim: int = 256,
        unrelated_latent_dim: int = 32,
        num_classes: int = 751,
        learning_rate: float = 0.001,
        related_coeff: float = 0.5,
        unrelated_coeff: float = 0.001,
        img_recon_coeff: float = 5,
        code_recon_coeff: float = 1,
        adv_coeff: float = 1,
        id_coeff: float = 0.5,
        finetuning_backbone: bool = False,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.__build_model()

        self.automatic_optimization = False

        self.criterion_recon = nn.L1Loss()
        self.criterion_adv = nn.BCEWithLogitsLoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_kld = nn.KLDivLoss(reduction="batchmean")

        self.metric_accuracy = Accuracy()
        self.metric_map = RetrievalMAP()

    def __build_model(self):
        self.backbone = ResnetBackbone(
            self.hparams.backbone_type, True, True
        )  # Loss: None
        self.related_encoder = RelatedEncoder(  # Loss: None
            self.backbone.out_features, self.hparams.related_latent_dim
        )
        self.unrelated_encoder = UnrelatedEncoder(  # Loss: KL Divergence
            self.backbone.out_features, self.hparams.unrelated_latent_dim
        )
        self.classifier = BasicClassifer(  # Loss: CrossEntropy
            self.hparams.num_classes, self.hparams.related_latent_dim
        )
        self.generator = ISGAN_Generator(  # Loss: Reconstruction
            self.hparams.related_latent_dim, self.hparams.unrelated_latent_dim
        )
        self.discriminator = Discriminator(
            self.hparams.num_classes
        )  # Loss: Adversarial, CrossEntropy

    def forward(self, x):
        related_features, unrelated_features = self.backbone(x)
        related_codes = self.related_encoder.encode(related_features)
        unrelated_codes = self.unrelated_encoder.encode(unrelated_features)
        return related_codes, unrelated_codes

    def encode(
        self, x: torch.Tensor
    ) -> Union[tuple[torch.Tensor, ...], torch.Tensor]:
        return self.forward(x)

    def generate(self, *features: torch.Tensor) -> torch.Tensor:
        return self.generator.generate(*features)

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate

        opt_gen = optim.Adam(
            (
                # list(self.backbone.parameters()) +
                list(self.related_encoder.parameters())
                + list(self.unrelated_encoder.parameters())
                + list(self.classifier.parameters())
                + list(self.generator.parameters())
            ),
            learning_rate,
        )
        opt_dis = optim.Adam(self.discriminator.parameters(), learning_rate)

        return [opt_gen, opt_dis]

    def training_step(self, batch, batch_idx):
        optims: tuple[optim.Adam, optim.Adam] = self.optimizers()
        opt_gen, opt_dis = optims

        opt_gen.zero_grad()

        (x_a, x_p, x_n), (
            y_a,
            y_p,
            y_n,
        ) = batch

        if self.hparams.finetuning_backbone:
            self.backbone.eval()
            with torch.no_grad():
                f_r_a, f_u_a = self.backbone(x_a)
                f_r_p, f_u_p = self.backbone(x_p)
                f_r_n, f_u_n = self.backbone(x_n)
        else:
            f_r_a, f_u_a = self.backbone(x_a)
            f_r_p, f_u_p = self.backbone(x_p)
            f_r_n, f_u_n = self.backbone(x_n)

        r_a = self.related_encoder(f_r_a)  # related_anchor
        r_p = self.related_encoder(f_r_p)
        r_n = self.related_encoder(f_r_n)
        u_a, p_a, q_a = self.unrelated_encoder(f_u_a)
        u_p, p_p, q_p = self.unrelated_encoder(f_u_p)
        u_n, p_n, q_n = self.unrelated_encoder(f_u_n)

        # L_{related}: Classification, CrossEntropy
        y_hat_r_a = self.classifier(r_a)
        y_hat_r_p = self.classifier(r_p)
        y_hat_r_n = self.classifier(r_n)
        loss_a = self.criterion_cls(y_hat_r_a, y_a)
        loss_p = self.criterion_cls(y_hat_r_p, y_p)
        loss_n = self.criterion_cls(y_hat_r_n, y_n)
        loss_related = (loss_a + loss_p + loss_n) / 3
        accuracy_a = self.metric_accuracy(y_hat_r_a, y_a)
        accuracy_p = self.metric_accuracy(y_hat_r_p, y_p)
        accuracy_n = self.metric_accuracy(y_hat_r_n, y_n)
        accuracy_related = (accuracy_a + accuracy_p + accuracy_n) / 3

        self.log("L_{R}", loss_related, prog_bar=True)
        self.log("accuracy", accuracy_related, prog_bar=True)

        # L_{unrelated}: KL Divergence
        loss_a: Tensor = torch.distributions.kl_divergence(q_a, p_a)
        loss_p: Tensor = torch.distributions.kl_divergence(q_p, p_p)
        loss_n: Tensor = torch.distributions.kl_divergence(q_n, p_n)
        loss_unrelated = (loss_a.mean() + loss_p.mean() + loss_n.mean()) / 3

        """
        Self Identity Generation(개념상 존재; Image Reconstruction)
        """
        # L^{sameimg}_{recon}: L1
        x_hat_a = self.generator(r_a, u_a)
        x_hat_p = self.generator(r_p, u_p)
        x_hat_n = self.generator(r_n, u_n)
        loss_a = self.criterion_recon(x_hat_a, x_a)
        loss_p = self.criterion_recon(x_hat_p, x_p)
        loss_n = self.criterion_recon(x_hat_n, x_n)
        loss_sameimg_recon = (loss_a + loss_p + loss_n) / 3

        # L^{diffimg}_{recon}: L1
        x_hat_ap = self.generator(r_a, u_p)
        x_hat_pa = self.generator(r_p, u_a)
        loss_ap = self.criterion_recon(x_hat_ap, x_a)
        loss_pa = self.criterion_recon(x_hat_pa, x_p)
        loss_diffimg_recon = (loss_ap + loss_pa) / 2

        """
        Cross Identity Generation(개념상 존재; Code Reconstruction)
        """
        x_hat_an = self.generator(r_a, u_n)
        x_hat_na = self.generator(r_n, u_a)

        if self.hparams.finetuning_backbone:
            self.backbone.eval()
            with torch.no_grad():
                f_r_an, f_u_an = self.backbone(x_hat_an)
                f_r_na, f_u_na = self.backbone(x_hat_na)
        else:
            f_r_an, f_u_an = self.backbone(x_hat_an)
            f_r_na, f_u_na = self.backbone(x_hat_na)

        # L^{related code}_{recon}: L1
        r_an = self.related_encoder(f_r_an)
        r_na = self.related_encoder(f_r_na)
        loss_an_a = self.criterion_recon(r_an, r_a)
        loss_na_n = self.criterion_recon(r_na, r_n)
        loss_relatedcode_recon = (loss_an_a + loss_na_n) / 2

        # L^{unrelated code}_{recon}: L1
        u_an, _, _ = self.unrelated_encoder(f_u_an)
        u_na, _, _ = self.unrelated_encoder(f_u_na)
        loss_an_n = self.criterion_recon(u_an, u_n)
        loss_na_a = self.criterion_recon(u_na, u_a)
        loss_unrelatedcode_recon = (loss_an_n + loss_na_a) / 2

        # L_{adv}: BCE
        """
        1. L^{sameimg}_{recon} 및 L^{diffimg}_{recon}과정에서 생성된 이미지를 D를 통해 판별 후 1이되도록 G 학습
        2. L^{sameimg}_{recon} 및 L^{diffimg}_{recon}과정에서 생성된 이미지를 D를 통해 판별 후 0이되도록 D 학습
        3. 실제 이미지를 D를 통해 판별 후 1이되도록 D 학습
        """
        # G 입장
        d_fake_a, _ = self.discriminator(x_hat_a)
        d_fake_p, _ = self.discriminator(x_hat_p)
        d_fake_n, _ = self.discriminator(x_hat_n)
        d_fake_ap, _ = self.discriminator(x_hat_ap)
        d_fake_pa, _ = self.discriminator(x_hat_pa)
        d_fake_an, _ = self.discriminator(x_hat_an)
        d_fake_na, _ = self.discriminator(x_hat_na)
        loss_adv_gen_a = self.criterion_adv(
            d_fake_a, torch.ones_like(d_fake_a)
        )
        loss_adv_gen_p = self.criterion_adv(
            d_fake_p, torch.ones_like(d_fake_p)
        )
        loss_adv_gen_n = self.criterion_adv(
            d_fake_n, torch.ones_like(d_fake_n)
        )
        loss_adv_gen_ap = self.criterion_adv(
            d_fake_ap, torch.ones_like(d_fake_ap)
        )
        loss_adv_gen_pa = self.criterion_adv(
            d_fake_pa, torch.ones_like(d_fake_pa)
        )
        loss_adv_gen_an = self.criterion_adv(
            d_fake_p, torch.ones_like(d_fake_an)
        )
        loss_adv_gen_na = self.criterion_adv(
            d_fake_n, torch.ones_like(d_fake_na)
        )
        loss_adv_gen = (
            loss_adv_gen_a
            + loss_adv_gen_p
            + loss_adv_gen_n
            + loss_adv_gen_ap
            + loss_adv_gen_pa
            + loss_adv_gen_an
            + loss_adv_gen_na
        ) / 7

        # G 학습 시작
        loss_gen_all = (
            (loss_related * self.hparams.related_coeff)
            + (loss_unrelated * self.hparams.unrelated_coeff)
            + (
                (loss_sameimg_recon + loss_diffimg_recon)
                / 2
                * self.hparams.img_recon_coeff
            )
            + (
                (loss_relatedcode_recon + loss_unrelatedcode_recon)
                / 2
                * self.hparams.code_recon_coeff
            )
            + (loss_adv_gen * self.hparams.adv_coeff)
        )
        self.manual_backward(loss_gen_all)
        opt_gen.step()

        # D 입장
        opt_dis.zero_grad()

        d_fake_a, _ = self.discriminator(x_hat_a.detach())
        d_fake_p, _ = self.discriminator(x_hat_p.detach())
        d_fake_n, _ = self.discriminator(x_hat_n.detach())
        d_fake_ap, _ = self.discriminator(x_hat_ap.detach())
        d_fake_pa, _ = self.discriminator(x_hat_pa.detach())
        d_fake_an, _ = self.discriminator(x_hat_an.detach())
        d_fake_na, _ = self.discriminator(x_hat_na.detach())

        loss_adv_dis_fake_a = self.criterion_adv(
            d_fake_a, torch.zeros_like(d_fake_a)
        )
        loss_adv_dis_fake_p = self.criterion_adv(
            d_fake_p, torch.zeros_like(d_fake_p)
        )
        loss_adv_dis_fake_n = self.criterion_adv(
            d_fake_n, torch.zeros_like(d_fake_n)
        )
        loss_adv_dis_fake_ap = self.criterion_adv(
            d_fake_ap, torch.zeros_like(d_fake_ap)
        )
        loss_adv_dis_fake_pa = self.criterion_adv(
            d_fake_pa, torch.zeros_like(d_fake_pa)
        )
        loss_adv_dis_fake_an = self.criterion_adv(
            d_fake_p, torch.zeros_like(d_fake_an)
        )
        loss_adv_dis_fake_na = self.criterion_adv(
            d_fake_n, torch.zeros_like(d_fake_na)
        )
        loss_adv_dis_fake = (
            loss_adv_dis_fake_a
            + loss_adv_dis_fake_p
            + loss_adv_dis_fake_n
            + loss_adv_dis_fake_ap
            + loss_adv_dis_fake_pa
            + loss_adv_dis_fake_an
            + loss_adv_dis_fake_na
        ) / 7

        d_real_a, y_hat_a = self.discriminator(x_a)
        d_real_p, y_hat_p = self.discriminator(x_p)
        d_real_n, y_hat_n = self.discriminator(x_n)
        loss_adv_dis_real_a = self.criterion_adv(
            d_real_a, torch.ones_like(d_real_a)
        )
        loss_adv_dis_real_n = self.criterion_adv(
            d_real_p, torch.ones_like(d_real_p)
        )
        loss_adv_dis_real_p = self.criterion_adv(
            d_real_n, torch.ones_like(d_real_n)
        )
        loss_adv_dis_real = (
            loss_adv_dis_real_a + loss_adv_dis_real_n + loss_adv_dis_real_p
        ) / 3

        # D 학습 시작
        loss_adv_dis = (loss_adv_dis_fake + loss_adv_dis_real) / 2

        # L_{id}: CE
        """
        1. 실제 이미지를 D를 통해 분류 후 D 학습
        """
        loss_id_a = self.criterion_cls(y_hat_a, y_a)
        loss_id_p = self.criterion_cls(y_hat_p, y_p)
        loss_id_n = self.criterion_cls(y_hat_n, y_n)
        loss_id = (loss_id_a + loss_id_p + loss_id_n) / 3

        loss_dis_all = (loss_adv_dis * self.hparams.adv_coeff) + (
            loss_id * self.hparams.id_coeff
        )

        self.manual_backward(loss_dis_all)
        opt_dis.step()

        self.log("L_{U}", loss_unrelated)
        self.log(
            "L^{img}_{recon}", (loss_sameimg_recon + loss_diffimg_recon) / 2
        )
        self.log(
            "L^{code}_{recon}",
            (loss_relatedcode_recon + loss_unrelatedcode_recon) / 2,
        )
        self.log("L_{adv} gen", loss_adv_gen)
        self.log("L_{adv} dis", loss_adv_dis)
