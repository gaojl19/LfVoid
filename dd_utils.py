import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
# from tqdm.notebook import tqdm
from tqdm import tqdm
import os
import math
import pdb


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            is_cross = encoder_hidden_states is not None
            
            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
        return forward
        
        # def forward(x, context=None, mask=None):
            
            # batch_size, sequence_length, dim = x.shape
            # h = self.heads
            
            # # x goes through the query projection
            # q = self.to_q(x)
            # is_cross = context is not None
            # context = context if is_cross else x 
            
            # # if cross attention, context goes through the key and value projection
            # # else, x goes through the key and value projection
            # k = self.to_k(context) 
            # v = self.to_v(context)
            
            # q = self.reshape_heads_to_batch_dim(q) 
            # k = self.reshape_heads_to_batch_dim(k)
            # v = self.reshape_heads_to_batch_dim(v)
            
            # # attention score of query and key
            # sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            
            # # mask
            # if mask is not None:
            #     mask = mask.reshape(batch_size, -1)
            #     max_neg_value = -torch.finfo(sim.dtype).max
            #     mask = mask[:, None, :].repeat(h, 1, 1)
            #     sim.masked_fill_(~mask, max_neg_value)
            
            # # attention probability
            # attn = sim.softmax(dim=-1)
            
            # # pass through controller to alter the attention map
            # attn = controller(attn, is_cross, place_in_unet)
            # # if is_cross:
            # #     print(place_in_unet, "cross", attn.shape)
            
            # # calculate attention with value states
            # out = torch.einsum("b i j, b j d -> b i d", attn, v)
            # out = self.reshape_batch_dim_to_heads(out)
            
            # # out
            # return to_out(out)

        # return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
            device=generator.device
        )
    
    # latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    # latent: [batch_size, 4, 64, 64]
    
    # BATCH: expand latent
    if batch_size > latent.shape[0]: # the target prompt is entered
        latents = latent.repeat(2, 1, 1, 1)
    else:
        latents = latent.repeat(1, 1, 1, 1)
        
    return latent, latents


def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    
    # latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]   
    latents = model.scheduler.step(noise_pred, t, latents).prev_sample 
    return latents



def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def save_images(images, num_rows=1, offset_ratio=0.02, tags="", save_dir="./outputs/", region=None, plot_region=False):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    
    if region and plot_region:
        pil_img = pil_img.resize((512, 512))
        image_editable = ImageDraw.Draw(pil_img)
        
        x0 = region[0] * 512
        y0 = region[2] * 512
        x1 = region[1] * 512
        y1 = region[3] * 512
        image_editable.rectangle(
            xy=[x0, y0, x1, y1], outline=(255, 0, 0, 255), width=8
        )
        pil_img.save(os.path.join(save_dir, tags + ".png"))
        return
        
    else:
        pil_img.save(os.path.join(save_dir, tags + ".png"))
        return # not plotting the region
    
        
    
    
    
def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    """ Gaussian weight
    Args:
       x(float): sample x
       x(float): sample x
    """
    return (
        1
        / (2 * math.pi * sx * sy)
        * torch.exp(-((x - mx) ** 2 / (2 * sx ** 2) + (y - my) ** 2 / (2 * sy ** 2)))
    )
    


        