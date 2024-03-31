import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
import torch.distributed as dist

from safetensors import safe_open
from tqdm import tqdm
from einops import rearrange
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, \
    convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora, load_diffusers_lora


# 这个函数的作用是为了在分布式环境中控制输出，只有 rank 为 0 的进程才会输出相关信息，这样可以避免在多个进程同时输出信息时产生混乱。
def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


# 将生成的视频逐帧处理，并且转换成gif格式
def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")  # TBCHW
    outputs = []
    for x in videos:  # BCHW 即逐帧处理
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # CHW  如果有多批，那么将他们排列成网格状。
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # CHW->HCW->HWC->HW
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


# DDIM Inversion
'''
    在某些生成模型中（如GPT-3等变种），加入无条件嵌入是为了提供一个模型可以从中开始生成内容的“空白”上下文。这里的“无条件”意味着不依赖于任何特
定的输入或提示信息，而是利用模型自身学习到的概率分布来生成内容。
    当我们将无条件嵌入与基于给定`prompt`的条件嵌入结合时，我们实际上是创建了一个混合的起始点，使得模型既能考虑全局的概率分布（通过无条件部分），
又能依据具体提示信息（通过条件部分）生成相关的内容。
    例如，在文本生成场景下，无条件嵌入可能帮助模型生成更加多样且自然的开头，而条件嵌入则确保了生成的内容会围绕着用户提供的提示展开。这种设计有助于实
现更好的可控性和灵活性，特别是在使用自回归模型进行文本续写或者根据提示创作时。
'''


# 装饰器，禁用梯度计算功能
@torch.no_grad()
def init_prompt(prompt, pipeline):
    # 使用pipeline的tokenizer对空字符串进行编码，添加最大长度的填充，使其长度达到模型的最大可接受长度
    # 返回的数据类型为PyTorch张量
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    # 将无条件部分的token id通过pipeline的text_encoder进行编码，得到对应的嵌入向量
    # 并将结果移动到pipeline指定的设备上（可能是GPU或CPU）
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]

    # 对给定的prompt字符串进行编码，同样添加最大长度填充，进行截断（如果超过最大长度）
    # 并返回PyTorch张量形式的编码结果
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # 将prompt编码后的token id通过pipeline的text_encoder获取对应的嵌入向量
    # 同样将结果移到pipeline指定的设备上
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    # 将无条件嵌入和基于prompt的条件嵌入按顺序拼接起来，形成一个综合的上下文向量
    context = torch.cat([uncond_embeddings, text_embeddings])
    # 返回这个上下文向量，它将作为后续生成过程的初始输入
    return context


# 定义函数next_step，该函数用于在DDIM逆向扩散过程中计算下一个迭代步骤的结果
def next_step(model_output: Union[torch.FloatTensor, np.ndarray],
              # 输入：模型在当前时间步预测的噪声修正量，支持torch.FloatTensor或np.ndarray类型
              timestep: int,  # 输入：当前时间步编号，int类型
              sample: Union[torch.FloatTensor, np.ndarray],  # 输入：当前噪声样本，支持torch.FloatTensor或np.ndarray类型
              ddim_scheduler):  # 输入：DDIM调度器对象，包含预计算的扩散过程参数

    # 根据DDIM逆向过程的步长策略，计算下一个有效的逆向时间步编号（不超过训练时的最大时间步数）
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep

    # 获取当前时间步对应的累积alpha值（clean signal probability accumulation）
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod

    # 获取下一个时间步对应的累积alpha值
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]

    # 计算当前时间步对应的累积beta值（noise probability accumulation），等于1 - alpha_prod_t
    beta_prod_t = 1 - alpha_prod_t

    # 计算从当前噪声样本还原出原始样本的一个估计值
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

    # 计算从当前时间步到下一时间步的噪声修正方向
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output

    # 结合原始样本估计值与噪声修正方向，计算下一个时间步的噪声样本
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction

    # 返回计算得出的下一个时间步的噪声样本
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def load_weights(
        animation_pipeline,
        # motion module
        motion_module_path="",
        motion_module_lora_configs=[],
        # domain adapter
        adapter_lora_path="",
        adapter_lora_scale=1.0,
        # image layers
        dreambooth_model_path="",
        lora_model_path="",
        lora_alpha=0.8,
):
    # motion module
    unet_state_dict = {}
    if motion_module_path != "":
        print(f"load motion module from {motion_module_path}")
        motion_module_state_dict = torch.load(motion_module_path, map_location="cpu")
        motion_module_state_dict = motion_module_state_dict[
            "state_dict"] if "state_dict" in motion_module_state_dict else motion_module_state_dict
        unet_state_dict.update(
            {name: param for name, param in motion_module_state_dict.items() if "motion_modules." in name})
        unet_state_dict.pop("animatediff_config", "")

    missing, unexpected = animation_pipeline.unet.load_state_dict(unet_state_dict, strict=False)
    assert len(unexpected) == 0
    del unet_state_dict

    # base model
    if dreambooth_model_path != "":
        print(f"load dreambooth model from {dreambooth_model_path}")
        if dreambooth_model_path.endswith(".safetensors"):
            dreambooth_state_dict = {}
            with safe_open(dreambooth_model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    dreambooth_state_dict[key] = f.get_tensor(key)
        elif dreambooth_model_path.endswith(".ckpt"):
            dreambooth_state_dict = torch.load(dreambooth_model_path, map_location="cpu")

        # 1. vae
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(dreambooth_state_dict, animation_pipeline.vae.config)
        animation_pipeline.vae.load_state_dict(converted_vae_checkpoint)
        # 2. unet
        converted_unet_checkpoint = convert_ldm_unet_checkpoint(dreambooth_state_dict, animation_pipeline.unet.config)
        animation_pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
        # 3. text_model
        animation_pipeline.text_encoder = convert_ldm_clip_checkpoint(dreambooth_state_dict)
        del dreambooth_state_dict

    # lora layers
    if lora_model_path != "":
        print(f"load lora model from {lora_model_path}")
        assert lora_model_path.endswith(".safetensors")
        lora_state_dict = {}
        with safe_open(lora_model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                lora_state_dict[key] = f.get_tensor(key)

        animation_pipeline = convert_lora(animation_pipeline, lora_state_dict, alpha=lora_alpha)
        del lora_state_dict

    # domain adapter lora
    if adapter_lora_path != "":
        print(f"load domain lora from {adapter_lora_path}")
        domain_lora_state_dict = torch.load(adapter_lora_path, map_location="cpu")
        domain_lora_state_dict = domain_lora_state_dict[
            "state_dict"] if "state_dict" in domain_lora_state_dict else domain_lora_state_dict
        domain_lora_state_dict.pop("animatediff_config", "")

        animation_pipeline = load_diffusers_lora(animation_pipeline, domain_lora_state_dict, alpha=adapter_lora_scale)

    # motion module lora
    for motion_module_lora_config in motion_module_lora_configs:
        path, alpha = motion_module_lora_config["path"], motion_module_lora_config["alpha"]
        print(f"load motion LoRA from {path}")
        motion_lora_state_dict = torch.load(path, map_location="cpu")
        motion_lora_state_dict = motion_lora_state_dict[
            "state_dict"] if "state_dict" in motion_lora_state_dict else motion_lora_state_dict
        motion_lora_state_dict.pop("animatediff_config", "")

        animation_pipeline = load_diffusers_lora(animation_pipeline, motion_lora_state_dict, alpha)

    return animation_pipeline
