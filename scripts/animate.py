import argparse
import datetime
import os
from omegaconf import OmegaConf
import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat
from pathlib import Path
from PIL import Image
import numpy as np


@torch.no_grad()  # 装饰器，禁用梯度计算
def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")  # 获取当前时间并且规范化格式
    savedir = f"samples/{Path(args.config).stem}-{time_str}"  # 获取配置路径里最后一个文件名称并与时间相连接
    os.makedirs(savedir)  # 创建目录文件夹

    '''
    OmegaConf.load(args.config) 会从指定的配置文件（通常是 YAML 或 JSON 格式）中加载配置，并将其转化为一个 OmegaConf 对象。
    这个对象可以直接像操作 Python 字典那样去访问和修改配置项，非常适合在程序运行时动态调整配置参数。
    '''
    config = OmegaConf.load(args.config)
    samples = []

    # create validation pipeline
    '''
    subfolder="tokenizer" 表示在预训练模型路径下寻找包含 tokenizer 权重的子文件夹。这是因为预训练模型通常会把模型权重和 tokenizer 权重分别存储在不同的子文件夹中。
    '''
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cuda()

    sample_idx = 0

    for model_idx, model_config in enumerate(config):

        model_config.W = model_config.get("W", args.W)  # 若model_config中没有W这个属性的值，那么就用后面的替代
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))

        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet",
                                                       unet_additional_kwargs=OmegaConf.to_container(
                                                           inference_config.unet_additional_kwargs)).cuda()
        # OmegaConf.to_container(...) 将配置信息转换成Python容器对象（如字典），以便作为关键字参数传入

        # load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""  # assert 断言，如果不成立，立刻终止执行
            assert model_config.get("controlnet_config", "") != ""

            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None  # unet配置

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get(
                "controlnet_additional_kwargs", {}))  # controlnet配置

            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")

            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")  # 加载controlnet参数

            controlnet_state_dict = controlnet_state_dict[
                "controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict

            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cuda()

            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]  # 将图片路径统一转换为列表类型

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L  # 防止因输入图片过多导致模型无法正确处理或者超出预期范围的问题

            '''
            - transforms.Compose 
               - 是 PyTorch 的 torchvision 库中的一个类，它用于组合多个图像预处理操作，形成一个完整的图像处理流水线。在训练深度学习模型时，经
               - 常需要对输入图像进行一系列预处理，如调整大小、裁剪、归一化等操作。通过使用 transforms.Compose，可以方便地将这些操作串联起来，一次性应用于每一张输入图像。
            - transforms.RandomResizedCrop(size, scale, ratio) 
               - size=(model_config.H, model_config.W) 指定了随机裁剪后图像的目标大小，高度为model_config.H，宽度为model_config.W。
               - scale=(1.0, 1.0) 表示随机缩放的比例范围，这里设置为1.0，意味着只会按目标尺寸裁剪，不会进行缩放。
               - ratio=(model_config.W / model_config.H, model_config.W / model_config.H) 表示裁剪区域的宽高比范围，这里设置为图像固定的宽高比，即裁剪出来的区域将保持与目标尺寸相同的宽高比。
            - transforms.ToTensor()
               - 这个操作将 PIL Image 或者 numpy.ndarray 类型的图像数据转换为 PyTorch Tensor，这是大多数PyTorch模型需要的输入格式。
            '''
            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0),
                    ratio=(model_config.W / model_config.H, model_config.W / model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):  # 归一化处理
                    """
                    - image.mean(dim=0, keepdim=True) 计算图像在通道维度（即第一个维度，dim=0）上的平均值，并通过 keepdim=True 参数保留维度大小，使得输出仍为三维张量，形状变为 (1, H, W)。
                    - .repeat(3, 1, 1) 将上一步得到的 (1, H, W) 形状的张量在通道维度上重复三次，得到 (3, H, W) 形状的张量，其内容为原来各个通道的平均值。
                    """
                    image = image.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                    # 接下来减去最小值，除以最大值使得图像数据的范围变成0-1之间
                    image -= image.min()
                    image /= image.max()
                    return image
            else:
                image_norm = lambda x: x

            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)

            """
            - image.numpy()：首先将张量 image 转换为NumPy数组。深度学习框架中的张量经常被用来表示和处理图像数据。
            - .transpose(1, 2, 0)：对数组进行转置，将原本的（通道，高度，宽度）格式转换为（高度，宽度，通道）格式，这是符合OpenCV和PIL等库对图像数据排列顺序的要求。
            - (255. * ...)：将数值范围从0-1扩展到0-255，因为大多数图像文件格式（包括PNG）使用的是0-255的整数像素值。
            - .astype(np.uint8)：将数据类型转换为无符号8位整数（np.uint8），这是图像像素常用的存储格式。
            - Image.fromarray(...)：使用PIL库的Image类从数组创建一个图像对象。
            - .save(f"{savedir}/control_images/{i}.png")：最后，将创建的图像对象保存为PNG格式的文件，存放在指定的目录savedir/control_images/下，并按照索引i命名图像文件名，例如0.png, 1.png等。
            """
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1, 2, 0))).astype(np.uint8)).save(
                    f"{savedir}/control_images/{i}.png")

            """
            - torch.stack(controlnet_images)：torch.stack() 是 PyTorch 中的一个函数，它将一个包含多个张量的列表 controlnet_images 按照新的维度（默认是最外层）堆
              叠成一个新的张量。这样，如果你有一系列经过处理和归一化的图像张量，它们会被组合成一个新的四维张量，其中第一维度是批量大小（在这个例子中，controlnet_images 列表中的图像数量）。
            - .unsqueeze(0)：unsqueeze() 方法会在张量的指定维度插入一个新的尺寸，使其大小为1。这里在第一个维度（批量维度）插入了新尺寸，将原本可能的三维张量（批次大小高度宽度
              通道数）变为四维张量（1批次大小高度宽度*通道数）。这对于某些深度学习模型来说是必要的，因为许多模型需要输入数据具有批量维度，即使当前只有一个样本。
            """
            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
            # 原本的images是三维张量，使用stack堆叠增加一维，使用unsqueeze再次增加一维，成为五维张量。
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")
            # 将controlnet_images张量的维度从(batch_size, frames, channels, height, width)
            # 重新排列为(batch_size, channels, frames, height, width)
            # 这里的frame是图片的数量，相当于帧数。

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]  # 获取frame
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                # (b f)：将批次维度b和帧维度f合并为一个新的批次维度，即每个批次的帧数现在作为新批次的一个单位
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                #  将输入图像数据归一化到[-1, 1]区间内；将归一化后的图像数据作为输入，生成对应的潜在变量分布；
                #  从编码器产生的潜在变量分布中采样得到一组潜在向量（latent vectors）。VAE的编码器并不直接输出单一的潜在向量，而是输出概率分布，然后从该分布中采样。
                # 这一步对采样的潜在向量进行缩放。这个比例系数可能是实验中发现的最佳缩放因子
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        # set xformers
        if is_xformers_available() and (not args.without_xformers):
            unet.enable_xformers_memory_efficient_attention()
            # 在UNet模型（或者其他集成XFormers库的模型）中启用一种高效的注意力机制——XFormers的内存高效注意力机制
            if controlnet is not None: controlnet.enable_xformers_memory_efficient_attention()

        """
        - DDIMScheduler:DDIMScheduler 是 Denoising Diffusion Implicit Models (DDIM) 的一种推断时间调度器。
        在基于扩散模型的生成任务中，它用于按照特定的时间步长策略逐步从随机噪声还原出清晰的图像或其他数据样本。
        - OmegaConf.to_container(inference_config.noise_scheduler_kwargs):OmegaConf 是一个用于处理配置数据的库，通常用于 Hydra 或类似的多配置环境。
        - inference_config 可能是一个包含模型推断过程中所有相关设置的配置对象，其中包含了与噪声调度器（即 DDIMScheduler）相关的参数。
        - noise_scheduler_kwargs 应该是指定了 DDIMScheduler 初始化所需的关键字参数的子集。
        - OmegaConf.to_container() 方法将这些配置对象转换为 Python 基础类型（如字典），这样就可以直接作为关键字参数传给 DDIMScheduler 类的初始化函数。
        - **是解包一个字典作为关键字参数
        """
        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cuda")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path=model_config.get("motion_module", ""),
            motion_module_lora_configs=model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path=model_config.get("adapter_lora_path", ""),
            adapter_lora_scale=model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path=model_config.get("dreambooth_path", ""),
            lora_model_path=model_config.get("lora_model_path", ""),
            lora_alpha=model_config.get("lora_alpha", 0.8),
        ).to("cuda")

        prompts = model_config.prompt
        n_prompts = list(model_config.n_prompt) * len(prompts) if len(
            model_config.n_prompt) == 1 else model_config.n_prompt
        # 将n_prompt重复n次

        random_seeds = model_config.get("seed", [-1])
        random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
        random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
        # 得到一个与 prompts 列表长度相匹配的随机种子列表

        config[model_idx].random_seed = []

        for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):

            # manually set random seed for reproduction
            if random_seed != -1:
                torch.manual_seed(random_seed)
            else:
                torch.seed()
            config[model_idx].random_seed.append(torch.initial_seed())

            print(f"current seed: {torch.initial_seed()}")
            print(f"sampling {prompt} ...")
            sample = pipeline(
                prompt,
                negative_prompt=n_prompt,
                num_inference_steps=model_config.steps,
                guidance_scale=model_config.guidance_scale,
                width=model_config.W,
                height=model_config.H,
                video_length=model_config.L,

                controlnet_images=controlnet_images,
                controlnet_image_index=model_config.get("controlnet_image_indexs", [0]),
            ).videos
            samples.append(sample)

            prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
            save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
            print(f"save to {savedir}/sample/{prompt}.gif")

            sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)

    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5", )
    parser.add_argument("--inference-config", type=str, default="configs/inference/inference-v1.yaml")
    parser.add_argument("--config", type=str, required=True)

    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    parser.add_argument("--without-xformers", action="store_true")

    args = parser.parse_args()
    main(args)
