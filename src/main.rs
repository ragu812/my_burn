use burn::backend::Wgpu;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::record::{Recorder, CompactRecorder};
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn::tensor::ElementConversion;

#[derive(Module, Debug)]
pub struct SimpleEncoder<B: Backend> {
    conv1: Conv2d<B>,
    bn_1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn_2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn_3: BatchNorm<B, 2>,
    conv4: Conv2d<B>,
    bn_4: BatchNorm<B, 2>,
    mean: Linear<B>,
    variance: Linear<B>,
    latent_dimen: usize,
}

impl<B: Backend> SimpleEncoder<B> {
    pub fn new(in_channels: usize, latent_dimen: usize, device: &B::Device) -> Self {
        Self {
            conv1: Conv2dConfig::new([in_channels, 64], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn_1: BatchNormConfig::new(64).init(device),

            conv2: Conv2dConfig::new([64, 128], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn_2: BatchNormConfig::new(128).init(device),

    conv3: Conv2dConfig::new([128, 256], [3, 3])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(1, 1))
        .init(device),
    bn_3: BatchNormConfig::new(256).init(device),

    conv4: Conv2dConfig::new([256, 512], [3, 3])
        .with_stride([2, 2])
        .with_padding(PaddingConfig2d::Explicit(1, 1))
        .init(device),
            bn_4: BatchNormConfig::new(512).init(device),

            mean: LinearConfig::new(512 * 8 * 8, latent_dimen * 16).init(device),
            variance: LinearConfig::new(512 * 8 * 8, latent_dimen * 16).init(device),
            latent_dimen,
        }
    }

    pub fn forward(&self, input1: Tensor<B, 4>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let x = self.bn_1.forward(self.conv1.forward(input1));
        let x = activation::relu(x);

        let x = self.bn_2.forward(self.conv2.forward(x));
        let x = activation::relu(x);

        let x = self.bn_3.forward(self.conv3.forward(x));
        let x = activation::relu(x);

        let x = self.bn_4.forward(self.conv4.forward(x));
        let x = activation::relu(x);

        let x = x.flatten(1, 3);

        let mean = self.mean.forward(x.clone());
        let variance = self.variance.forward(x.clone());

        (mean, variance)
    }
}

#[derive(Module, Debug)]
pub struct SimpleDecoder<B: Backend> {
    transfer: Linear<B>,
    reverse1: ConvTranspose2d<B>,
    bn_1: BatchNorm<B, 2>,
    reverse2: ConvTranspose2d<B>,
    bn_2: BatchNorm<B, 2>,
    reverse3: ConvTranspose2d<B>,
    bn_3: BatchNorm<B, 2>,
    reverse4: ConvTranspose2d<B>,
    latent_dimen: usize,
    output: usize,
}

impl<B: Backend> SimpleDecoder<B> {
    pub fn new(output: usize, latent_dimen: usize, device: &B::Device) -> Self {
        Self {
            transfer: LinearConfig::new(latent_dimen * 16, 512 * 8 * 8).init(device),

            reverse1: ConvTranspose2dConfig::new([512, 256], [3, 3])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            bn_1: BatchNormConfig::new(256).init(device),

            reverse2: ConvTranspose2dConfig::new([256, 128], [3, 3])
                .with_stride([2, 2])
                .with_padding([1, 1])
                .init(device),
            bn_2: BatchNormConfig::new(128).init(device),

            reverse3: ConvTranspose2dConfig::new([128, 64], [3, 3])
                .with_stride([1, 1])
                .with_padding([1, 1])
                .init(device),
            bn_3: BatchNormConfig::new(64).init(device),

            reverse4: ConvTranspose2dConfig::new([64, output], [3, 3])
                .with_stride([1, 1])
                .with_padding([1, 1])
                .init(device),

            latent_dimen,
            output,
        }
    }

    pub fn forward(&self, input2: Tensor<B, 4>) -> Tensor<B, 4> {
        let input2_flat: Tensor<B, 2> = input2.flatten(1, 3);
        let y = self.transfer.forward(input2_flat);
        let y = activation::relu(y);
        let batch_size = y.dims()[0];
        let y = y.reshape([batch_size, 512, 8, 8]);

        let y = self.reverse1.forward(y);
        let y = self.bn_1.forward(y);
        let y = activation::relu(y);
        let y = self.reverse2.forward(y);
        let y = self.bn_2.forward(y);
        let y = activation::relu(y);

        let y = self.reverse3.forward(y);
        let y = self.bn_3.forward(y);
        let y = activation::relu(y);

        let y = self.reverse4.forward(y);
        activation::tanh(y)
    }
}

#[derive(Module, Debug)]
pub struct Vae<B: Backend> {
    encoder: SimpleEncoder<B>,
    decoder: SimpleDecoder<B>,
}

impl<B: Backend> Vae<B> {
    pub fn new(in_channels: usize, latent_dimen: usize, device: &B::Device) -> Self {
        Self {
            encoder: SimpleEncoder::new(in_channels, latent_dimen, device),
            decoder: SimpleDecoder::new(in_channels, latent_dimen, device),
        }
    }

    pub fn parametrize(&self, mean: Tensor<B, 2>, variance: Tensor<B, 2>) -> Tensor<B, 4> {
        let std = (variance.clone() * 0.5).exp();
        let noise = Tensor::random_like(&std, burn::tensor::Distribution::Normal(0.0, 1.0));
        let z = mean + noise * std;
        z.clone().reshape([z.dims()[0], self.encoder.latent_dimen, 4, 4])
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 4>, Tensor<B, 2>, Tensor<B, 2>) {
        let (mean, variance) = self.encoder.forward(x);
        let z = self.parametrize(mean.clone(), variance.clone());
        let reconstruction = self.decoder.forward(z);
        (reconstruction, mean, variance)
    }

    pub fn encode(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (mean, variance) = self.encoder.forward(x);
        self.parametrize(mean, variance)
    }

    pub fn decode(&self, y: Tensor<B, 4>) -> Tensor<B, 4> {
        self.decoder.forward(y)
    }

    pub fn vae_loss(
        mean: Tensor<B, 2>,
        variance: Tensor<B, 2>,
        reconstruction: Tensor<B, 4>,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let epsilon = 1e-8;
        let variance = variance + epsilon;
        let kl_loss = -0.5
            * (Tensor::ones_like(&variance) + variance.clone().log() - mean.powf_scalar(2.0)
                - variance.exp())
            .sum();
        let recon_flat: Tensor<B, 2> = reconstruction.flatten(1, 3);
        let input_flat: Tensor<B, 2> = input.flatten(1, 3);
        let recon_loss = burn::nn::loss::MseLoss::new().forward(
            recon_flat,
            input_flat,
            burn::nn::loss::Reduction::Mean,
        );
        recon_loss + kl_loss
    }
}

#[derive(Module, Debug)]
pub struct TimeAddition<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
}

impl<B: Backend> TimeAddition<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(dim, dim * 4).init(device),
            linear2: LinearConfig::new(dim * 4, dim).init(device),
        }
    }

    pub fn forward(&self, t: Tensor<B, 2>) -> Tensor<B, 2> {
        let z = self.linear1.forward(t);
        let z = activation::relu(z);
        self.linear2.forward(z)
    }
}

pub fn sin_time_addition<B: Backend>(
    device: &B::Device,
    time: Tensor<B, 1, Int>,
    dim: usize,
) -> Tensor<B, 2> {
    let batch_size = time.dims()[0];
    let half_dim = dim / 2;
    let t = time.float();

    let frequencies: Vec<f32> = (0..half_dim)
        .map(|i| {
            let exp = (i as f32) * 4.0 * std::f32::consts::LN_10 / (half_dim as f32);
            (-exp).exp()
        })
        .collect();

    let freqs: Tensor<B, 2> = Tensor::<B, 1>::from_floats(frequencies.as_slice(), device)
        .reshape([1, half_dim])
        .repeat_dim(0, batch_size);

    let arg = t.reshape([batch_size, 1]).repeat_dim(1, half_dim) * freqs;

    let sin_emb = arg.clone().sin();
    let cos_emb = arg.cos();

    Tensor::cat(vec![sin_emb, cos_emb], 1)
}

#[derive(Module, Debug)]
pub struct ResidualBlock<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    bn2: BatchNorm<B, 2>,
    time: Linear<B>,
}

impl<B: Backend> ResidualBlock<B> {
    pub fn new(device: &B::Device, channels: usize, time_emb_dim: usize) -> Self {
        Self {
            conv1: Conv2dConfig::new([channels, channels], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn1: BatchNormConfig::new(channels).init(device),

            conv2: Conv2dConfig::new([channels, channels], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            bn2: BatchNormConfig::new(channels).init(device),

            time: LinearConfig::new(time_emb_dim, channels).init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, t_emb: Tensor<B, 2>) -> Tensor<B, 4> {
        let h = self.conv1.forward(x.clone());
        let h = self.bn1.forward(h);
        let h = activation::silu(h);

        let t = self.time.forward(t_emb);
        let t = activation::silu(t);
        let batch_size = t.dims()[0];
        let channels = t.dims()[1];
        let t = t.reshape([batch_size, channels, 1, 1]);
        let h = h + t;

        let h = self.conv2.forward(h);
        let h = self.bn2.forward(h);
        activation::silu(h + x)
    }
}

#[derive(Module, Debug)]
pub struct Unet<B: Backend> {
    down1: Conv2d<B>,
    down_res1: ResidualBlock<B>,
    down2: Conv2d<B>,
    down_res2: ResidualBlock<B>,
    down3: Conv2d<B>,
    down_res3: ResidualBlock<B>,
    down4: Conv2d<B>,
    down_res4: ResidualBlock<B>,
    down5: Conv2d<B>,
    down_res5: ResidualBlock<B>,
    mid_res1: ResidualBlock<B>,
    mid_res2: ResidualBlock<B>,
    mid_res3: ResidualBlock<B>,
    up1: ConvTranspose2d<B>,
    up_res1: ResidualBlock<B>,
    up2: ConvTranspose2d<B>,
    up_res2: ResidualBlock<B>,
    up3: ConvTranspose2d<B>,
    up_res3: ResidualBlock<B>,
    up4: ConvTranspose2d<B>,
    up_res4: ResidualBlock<B>,
    out_conv: Conv2d<B>,
    time_emb: TimeAddition<B>,
}

impl<B: Backend> Unet<B> {
    pub fn new(device: &B::Device, in_channels: usize, time_emb_dim: usize) -> Self {
        Self {
            time_emb: TimeAddition::new(time_emb_dim, device),
            down1: Conv2dConfig::new([in_channels, 64], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            down_res1: ResidualBlock::new(device, 64, time_emb_dim),
            down2: Conv2dConfig::new([64, 128], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            down_res2: ResidualBlock::new(device, 128, time_emb_dim),
            down3: Conv2dConfig::new([128, 256], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            down_res3: ResidualBlock::new(device, 256, time_emb_dim),
            down4: Conv2dConfig::new([256, 512], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            down_res4: ResidualBlock::new(device, 512, time_emb_dim),
            down5: Conv2dConfig::new([512, 1024], [3, 3])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            down_res5: ResidualBlock::new(device, 1024, time_emb_dim),

            mid_res1: ResidualBlock::new(device, 1024, time_emb_dim),
            mid_res2: ResidualBlock::new(device, 1024, time_emb_dim),
            mid_res3: ResidualBlock::new(device, 1024, time_emb_dim),

            up1: ConvTranspose2dConfig::new([1024, 512], [3, 3])
                .with_stride([1, 1])
                .with_padding([1, 1])
                .init(device),
            up_res1: ResidualBlock::new(device, 512, time_emb_dim),
            up2: ConvTranspose2dConfig::new([512, 256], [3, 3])
                .with_stride([1, 1])
                .with_padding([1, 1])
                .init(device),
            up_res2: ResidualBlock::new(device, 256, time_emb_dim),
            up3: ConvTranspose2dConfig::new([256, 128], [3, 3])
                .with_stride([1, 1])
                .with_padding([1, 1])
                .init(device),          
            up_res3: ResidualBlock::new(device, 128, time_emb_dim),
            up4: ConvTranspose2dConfig::new([128, 64], [3, 3])
                .with_stride([1, 1])
                .with_padding([1, 1])
                .init(device),
            up_res4: ResidualBlock::new(device, 64, time_emb_dim),
            out_conv: Conv2dConfig::new([64, in_channels], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>, time: Tensor<B, 2>) -> Tensor<B, 4> {
        let time_embed = self.time_emb.forward(time);

        let down_1 = self.down1.forward(x);
        let down_1 = self.down_res1.forward(down_1, time_embed.clone());

        let down_2 = self.down2.forward(down_1.clone());
        let down_2 = self.down_res2.forward(down_2, time_embed.clone());

        let down_3 = self.down3.forward(down_2.clone());
        let down_3 = self.down_res3.forward(down_3, time_embed.clone());

        let down_4 = self.down4.forward(down_3.clone());
        let down_4 = self.down_res4.forward(down_4, time_embed.clone());

        let down_5 = self.down5.forward(down_4.clone());
        let down_5 = self.down_res5.forward(down_5, time_embed.clone());

        let mut m = self.mid_res1.forward(down_5, time_embed.clone());
        m = self.mid_res2.forward(m, time_embed.clone());
        m = self.mid_res3.forward(m, time_embed.clone());

        let u1 = self.up1.forward(m);
        let u1 = self.up_res1.forward(u1 + down_4, time_embed.clone());

        let u2 = self.up2.forward(u1.clone());
        let u2 = self.up_res2.forward(u2 + down_3, time_embed.clone());

        let u3 = self.up3.forward(u2.clone());
        let u3 = self.up_res3.forward(u3 + down_2, time_embed.clone());

        let u4 = self.up4.forward(u3.clone());
        let u4 = self.up_res4.forward(u4 + down_1, time_embed.clone());

        self.out_conv.forward(u4)
    }
}

#[derive(Module, Debug)]
pub struct DiffusionModel<B: Backend> {
    unet: Unet<B>,
    vae: Vae<B>,
    num_time: usize,
    latent_dimen: usize,
}

impl<B: Backend> DiffusionModel<B> {
    pub fn new(
        device: &B::Device,
        latent_dimen: usize,
        in_channels: usize,
        num_time: usize,
    ) -> Self {
        Self {
            vae: Vae::new(in_channels, latent_dimen, device),
            unet: Unet::new(device, latent_dimen, 256),
            num_time,
            latent_dimen,
        }
    }

    pub fn get_betas(&self, device: &B::Device) -> Tensor<B, 1> {
        let beta_start = 0.0001;
        let beta_end = 0.02;

        let steps = self.num_time;

        let betas: Vec<f32> = (0..steps)
            .map(|i| {
                beta_start + (beta_end - beta_start) * (i as f32) / ((steps - 1) as f32)
            })
            .collect();

        Tensor::from_floats(betas.as_slice(), device)
    }

    pub fn get_alphas(&self, device: &B::Device) -> Tensor<B, 1> {
        let betas = self.get_betas(device);
        let alphas = Tensor::ones_like(&betas) - betas;

        let alpha_vals: Vec<f32> = alphas.to_data().to_vec().unwrap();
        let mut cumulative = 1.0;
        let alpha_bars: Vec<f32> = alpha_vals
            .iter()
            .map(|&a| {
                cumulative *= a;
                cumulative
            })
            .collect();

        Tensor::from_floats(alpha_bars.as_slice(), device)
    }

    pub fn add_noise(
        &self,
        x: Tensor<B, 4>,
        time: Tensor<B, 1, Int>,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let alpha = self.get_alphas(device);
        let noise = Tensor::random_like(&x, burn::tensor::Distribution::Normal(0.0, 1.0));

        let batch_size = x.dims()[0];
        let channels = x.dims()[1];
        let height = x.dims()[2];
        let width = x.dims()[3];

        let mut noisy_samples = Vec::new();

        for i in 0..batch_size {
            let time_t = time.clone().slice([i..i + 1]).into_scalar().elem::<i32>() as usize;

            let alpha_t = alpha.clone().slice([time_t..time_t + 1]);

            let sqrt_alpha = alpha_t.clone().sqrt();
            let sqrt_one_minus_alpha = (Tensor::ones_like(&alpha_t) - alpha_t).sqrt();

            let xi = x.clone().slice([i..i + 1, 0..channels, 0..height, 0..width]);
            let noise_i = noise.clone().slice([i..i + 1, 0..channels, 0..height, 0..width]);

            let sqrt_alpha_exp = sqrt_alpha.clone().expand([1, channels, height, width]);
            let sqrt_one_minus_alpha_exp = sqrt_one_minus_alpha.clone().expand([1, channels, height, width]);
            let noisy_i = xi * sqrt_alpha_exp + noise_i * sqrt_one_minus_alpha_exp;

            noisy_samples.push(noisy_i);
        }

        let noisy = Tensor::cat(noisy_samples, 0);
        (noisy, noise)
    }

    pub fn noise_predict(&self, x_t: Tensor<B, 4>, time_emb: Tensor<B, 2>) -> Tensor<B, 4> {
        self.unet.forward(x_t, time_emb)
    }

    pub fn forward(&self, x: Tensor<B, 4>, device: &B::Device) -> Tensor<B, 1> {
        let batch_size = x.dims()[0];
        let z = self.vae.encode(x.clone());

        let t_values: Vec<i32> = (0..batch_size)
            .map(|_| (rand::random::<u32>() % self.num_time as u32) as i32)
            .collect();

        let t = Tensor::from_ints(t_values.as_slice(), device);
        let (z_noisy, noise) = self.add_noise(z, t.clone(), device);

        let t_emb = sin_time_addition(device, t, 256);

        let noise_pred = self.noise_predict(z_noisy, t_emb);
        let noise_pred_2d: Tensor<B, 2> = noise_pred.flatten(1, 3);

        let noise_2d: Tensor<B, 2> = noise.flatten(1, 3);

        (noise_pred_2d - noise_2d).powf_scalar(2.0).mean()
    }

    pub fn sample(
        &self,
        latent_dimen: usize,
        batch_size: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let mut z: Tensor<B, 4> = Tensor::random(
            [batch_size, latent_dimen, 4, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            device,
        );

        let beta = self.get_betas(device);
        let alpha_1 = self.get_alphas(device);

        for i in (0..self.num_time).rev() {
            let i_tensor = Tensor::from_ints(vec![i as i32; batch_size].as_slice(), device);
            let i_emb = sin_time_addition(device, i_tensor, 256);

            let noise_pred = self.noise_predict(z.clone(), i_emb);
            let noise_pred_2d: Tensor<B, 2> = noise_pred.flatten(1, 3);

            let alpha_t = alpha_1.clone().slice([i..i + 1]);
            let beta_t = beta.clone().slice([i..i + 1]);

            let alpha_sqrt = alpha_t.clone().sqrt();
            let alpha_sqrt_1 = (Tensor::ones_like(&alpha_t) - alpha_t.clone()).sqrt();

            let z_2d: Tensor<B, 2> = z.clone().flatten(1, 3);
            let alpha_sqrt_1_exp = alpha_sqrt_1.expand([latent_dimen * 16]).unsqueeze();
            let alpha_sqrt_exp = alpha_sqrt.expand([latent_dimen * 16]).unsqueeze();
            let pred_x0_2d = (z_2d - noise_pred_2d.clone() * alpha_sqrt_1_exp) / alpha_sqrt_exp;
            let pred_x0 = pred_x0_2d.clone().reshape([batch_size, latent_dimen, 4, 4]);

            if i > 0 {
                let alpha_prev = alpha_1.clone().slice([i - 1..i]);
                let alpha_prev_sqrt = alpha_prev.clone().sqrt();
                let alpha_prev_1 = (Tensor::ones_like(&alpha_prev) - alpha_prev).sqrt();

                let alpha_prev_sqrt_exp = alpha_prev_sqrt.expand([latent_dimen * 16]).unsqueeze();
                let alpha_prev_1_exp = alpha_prev_1.expand([latent_dimen * 16]).unsqueeze();
                let z_new_2d: Tensor<B, 2> = alpha_prev_sqrt_exp * pred_x0_2d.clone() + alpha_prev_1_exp * noise_pred_2d;
                z = z_new_2d.reshape([batch_size, latent_dimen, 4, 4]);

                let noise = Tensor::random_like(&z, burn::tensor::Distribution::Normal(0.0, 1.0));
                let beta_t_sqrt_exp = beta_t.sqrt().expand([latent_dimen, 4, 4]).unsqueeze();
                z = z + beta_t_sqrt_exp * noise;
            } else {
                z = pred_x0;
            }
        }
        self.vae.decode(z)
    }
}

pub fn save_model<B: Backend>(
    model: &DiffusionModel<B>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let recorder = CompactRecorder::new();
    recorder.record(model.clone().into_record(), path.into())?;
    println!("Model saved to {}", path);
    Ok(())
}

pub fn load_model<B: Backend>(
    path: &str,
    in_channels: usize,
    latent_dimen: usize,
    num_time: usize,
    device: &B::Device,
) -> Result<DiffusionModel<B>, Box<dyn std::error::Error>> {
    let recorder = CompactRecorder::new();
    let record = recorder.load(path.into(), device)?;

    let model = DiffusionModel::new(device, latent_dimen, in_channels, num_time).load_record(record);

    Ok(model)
}

use image::imageops::FilterType;
use std::fs;
use std::path::{Path, PathBuf};

pub struct Image {
    pub size: usize,
    pub channels: usize,
    pub image_paths: Vec<PathBuf>,
    pub height: usize,
    pub width: usize,
}

impl Image {
    pub fn directory(
        dir_path: &str,
        height: usize,
        width: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let path = Path::new(dir_path);

        if !path.exists() {
            return Err(format!("Directory does not exist: {}", dir_path).into());
        }
        let mut image_paths = Vec::new();

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(ext) = path.extension() {
                let ext = ext.to_str().unwrap_or("").to_lowercase();
                if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif") {
                    image_paths.push(path);
                }
            }
        }

        if image_paths.is_empty() {
            return Err(format!("No images found in directory: {}", dir_path).into());
        }

        image_paths.sort();
        let size = image_paths.len();

        println!("Loaded {} images from {}", size, dir_path);

        Ok(Self {
            image_paths,
            size,
            channels: 3,
            height,
            width,
        })
    }

    pub fn from_paths(
        paths: Vec<String>,
        height: usize,
        width: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut image_paths = Vec::new();

        for path_str in paths {
            let path = PathBuf::from(path_str);
            if !path.exists() {
                println!("Warning: File does not exist: {:?}", path);
                continue;
            }
            image_paths.push(path);
        }

        if image_paths.is_empty() {
            return Err("No valid image paths provided".into());
        }

        let size = image_paths.len();

        Ok(Self {
            image_paths,
            size,
            channels: 3,
            height,
            width,
        })
    }

    pub fn get<B: Backend>(
        &self,
        idx: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
        if idx >= self.size {
            return Err(format!("Index {} out of bounds (size: {})", idx, self.size).into());
        }

        let img = image::open(&self.image_paths[idx])?;
        let img = img.resize_exact(self.width as u32, self.height as u32, FilterType::Lanczos3);
        let img = img.to_rgb8();

        let mut data = Vec::with_capacity(self.channels * self.height * self.width);

        for pixel in img.pixels() {
            data.push((pixel[0] as f32 / 255.0) * 2.0 - 1.0);
            data.push((pixel[1] as f32 / 255.0) * 2.0 - 1.0);
            data.push((pixel[2] as f32 / 255.0) * 2.0 - 1.0);
        }

        let tensor: Tensor<B, 4> = Tensor::<B, 1>::from_floats(data.as_slice(), device)
            .reshape([1, self.channels, self.height, self.width]);
        Ok(tensor)
    }

    pub fn get_batch<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
        let mut batch_images = Vec::new();

        for &idx in indices {
            batch_images.push(self.get::<B>(idx, device)?);
        }

        Ok(Tensor::cat(batch_images, 0))
    }
}

pub fn train_ldm_epoch<B: Backend>(
    model: &DiffusionModel<B>,
    dataset: &Image,
    device: &B::Device,
    batch_size: usize,
    _learning_rate: f64,
) -> f32 {
    let mut total_loss = 0.0f32;
    let num_batches = dataset.size / batch_size;

    for batch_idx in 0..num_batches {
        let mut batch_images = Vec::new();

        for i in 0..batch_size {
            let idx = batch_idx * batch_size + i;
            if let Ok(img) = dataset.get::<B>(idx, device) {
                batch_images.push(img);
            }
        }

        if !batch_images.is_empty() {
            let images = Tensor::cat(batch_images, 0);
            let loss = model.forward(images, device);

            total_loss += loss.into_scalar().elem::<f32>();
        }
    }

    total_loss / num_batches as f32
}

fn main() {
    use burn::backend::{Autodiff, Wgpu};

    type Backend = Autodiff<Wgpu<f32>>;
    let device = Default::default();

    let in_channels = 3;
    let latent_dim = 256;
    let num_timesteps = 2000;
    let batch_size = 1;
    let num_epochs = 25;

    let mut model = DiffusionModel::<Backend>::new(&device, latent_dim, in_channels, num_timesteps);
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init::<Backend, DiffusionModel<Backend>>();

    let dataset = Image::directory(r"Pictures", 32, 32).unwrap();

    println!("Starting training...");
    for epoch in 0..num_epochs {
        let mut total_loss = 0.0f32;
        let num_batches = dataset.size / batch_size;

        println!("Epoch: {}", epoch);

        for batch_idx in 0..num_batches{
            let mut batch_images = Vec::new();

            println!("Batch: {}", batch_idx);

            for i in 0..batch_size {
                let idx = batch_idx * batch_size + i;
                if let Ok(img) = dataset.get::<Backend>(idx, &device) {
                    batch_images.push(img);
                }
            }

            if !batch_images.is_empty() {
                let images = Tensor::cat(batch_images, 0);
                let loss = model.forward(images, &device);

                let grads = loss.backward();
                let grad_params = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(1e-4, model, grad_params);
                total_loss += loss.into_scalar().elem::<f32>();
            }
        }

        let avg_loss = total_loss / num_batches as f32;
        println!("Epoch {}: Loss = {:.4}", epoch + 1, avg_loss);
    }

    save_model(&model, "trained_model.bin").unwrap();

    println!("Generating image");
    // Generate and save an image
    let generated = model.sample(latent_dim, 1, &device);
    let generated_img: burn::tensor::Tensor<burn::backend::Autodiff<burn::backend::NdArray<f32>>, 3> = generated.squeeze(0); // Remove batch dim
    let data: Vec<f32> = generated_img.to_data().to_vec().unwrap();
    let img = image::RgbImage::from_fn(32, 32, |x, y| {
        let idx = (y * 32 + x) as usize;
        let r_idx = idx * 3;
        let g_idx = idx * 3 + 1;
        let b_idx = idx * 3 + 2;
        image::Rgb([
            ((data[r_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
            ((data[g_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
            ((data[b_idx] * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0) as u8,
        ])
    });
    img.save("generated_image.png").unwrap();
    println!("Generated image saved to generated_image.png");
}