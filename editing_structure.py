## Null-text Directed Diffusion

from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import dd_utils
from torch.optim.adam import Adam
from PIL import Image
import os
import argparse
import copy


scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
noise_weight = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=10,
)
LOG_DIFFUSION = False
MY_TOKEN = ''
LOW_RESOURCE = False 
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
HEIGHT = 512
WIDTH = 512
DEBUG = False
device = None
ldm_stable = None
tokenizer = None
debug_path = "./outputs/debug_ptpdd"


try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")



class EmptyControl:
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
                
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
        
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn.detach().cpu()) # save original attention
        
        return attn
    

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention
    

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        
        
        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2: # replace smaller attention
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
        
        # return attn_base
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    
    def get_empty_edited_store(self):
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}
        
    
    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
            self.edited_attention_store = self.edited_step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        
        if DEBUG:
            self.plot_step_attention(edited=False)
            self.plot_step_attention(edited=True)
        
        self.step_store = self.get_empty_store()
        self.edited_step_store = self.get_empty_edited_store()
        
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super().forward(attn, is_cross, place_in_unet)
        h = attn.shape[0] // (self.batch_size)
        attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
        
        attn_base, attn_replace = copy.deepcopy(attn[0]), attn[1:]
         
        if is_cross and self.cur_step < self.cross_editing_steps:
            attn_replace, token_idx = self.replace_cross_attention(attn_base, attn_replace, place_in_unet)
            attn_replace_new = attn_base
            
            attn_replace_new[:, :, token_idx] = attn_replace[:, :, token_idx]
            attn[1:] = attn_replace_new
                
        elif self.cur_step < self.self_editing_steps:
            attn[1:] = self.replace_self_attention(attn_base, attn_replace, place_in_unet)
        
        attn = attn.reshape(self.batch_size * h, *attn.shape[2:])   
        return attn
    
    def save_edited_attn(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.edited_step_store[key].append(attn)
         
        return attn
    
    def save_attn(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn
    
    def get_average_edited_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.edited_attention_store[key]] for key in self.edited_attention_store}
        return average_attention
    
    
    def plot_step_attention(self, edited:bool):
        select = 1 if edited else 0
        
        tokens = tokenizer.encode(self.prompts[select])
        decoder = tokenizer.decode
        num_pixels = self.res ** 2
        
        attention_maps = []
        for location in self.from_where:
            # only visualize cross attention
            if edited:
                for item in self.step_store[f"{location}_{'cross'}"]:
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(len(self.prompts), -1, self.res, self.res, item.shape[-1])[select]
                        attention_maps.append(cross_maps)
            else:
                 for item in self.step_store[f"{location}_{'cross'}"]:
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(len(self.prompts), -1, self.res, self.res, item.shape[-1])[select]
                        attention_maps.append(cross_maps)
        
        attention_maps = torch.cat(attention_maps, dim=0)
        attention_maps = attention_maps.sum(0) / attention_maps.shape[0]
        
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.cpu().numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = dd_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        
        if edited:
            dd_utils.save_images(images=np.stack(images, axis=0), tags=f"step_{self.cur_step}", save_dir=self.edited_attention_out_path)
        else:
            dd_utils.save_images(images=np.stack(images, axis=0), tags=f"step_{self.cur_step}", save_dir=self.attention_out_path)
        
        
    def reset(self):
        super(AttentionControlEdit, self).reset()
        self.edited_attention_store = {}
        self.edited_step_store = self.get_empty_edited_store()
        
     
    def __init__(self, prompts, cross_editing_steps: int, self_editing_steps: int, debug_kwargs: dict = {}):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_editing_steps = cross_editing_steps # cross editing steps
        self.self_editing_steps = self_editing_steps # self editing steps
        self.edited_attention_store = {}
        self.edited_step_store = self.get_empty_edited_store()
        
        self.from_where = ["down", "up"]
        self.res = 16
        self.select = 0 # which diffusion process to visualize
        
        
        if DEBUG:
            self.attention_out_path = os.path.join(debug_path, "original")
            os.makedirs(self.attention_out_path, exist_ok=True)
            self.edited_attention_out_path = os.path.join(debug_path, "edited")
            os.makedirs(self.edited_attention_out_path, exist_ok=True)
            
            
class DirectedAttention(AttentionControlEdit):
    def __init__(self, prompts, cross_editing_steps: float, self_editing_steps: int, attn_kwargs: dict, debug_kwargs: dict):
        
        super(DirectedAttention, self).__init__(prompts, cross_editing_steps, self_editing_steps, debug_kwargs)
        self.prompts = prompts
        self.region = attn_kwargs["region"]
        self.prompt_token_idx = copy.deepcopy(attn_kwargs["token_idx"])
        self.token_idx = attn_kwargs["token_idx"]
        self.noise_scale = attn_kwargs["noise_scale"]
        self.num_trailing_maps = attn_kwargs["num_trailing_maps"]
        self.max_attn_size = attn_kwargs["max_attn_size"]
        self.annealing_coef = attn_kwargs["annealing_coef"]
        self.annealing_threshold = attn_kwargs["annealing_threshold"]
        self.use_base_edit = attn_kwargs["use_base_edit"]
        
        
    def replace_cross_attention(self, attn_base, attn_replace, place_in_unet):
        
        dim = int(np.sqrt(attn_base.size()[1]))
        
        if args.use_base_edit:
            attn_replace = attn_base
        
        # only edit the smaller attention maps, according to the map-size threshold
        if dim * dim > self.max_attn_size:
            return attn_replace, self.token_idx
        
        attn_replace = attn_replace.view(8, dim, dim, 77)
        
        if self.annealing_threshold > 0:
            positive_indices = torch.where(attn_replace < self.annealing_threshold)
            positive_mask = torch.zeros_like(attn_replace, dtype=torch.bool)
            positive_mask[positive_indices] = True
            
        global_mask = torch.zeros_like(attn_replace, dtype=torch.bool)
    
        # only support 1 region for now
        left = int(dim * self.region[0])
        right = int(dim * self.region[1])
        top = int(dim * self.region[2])
        bottom = int(dim * self.region[3])
        
        tmp = attn_replace[:, top:bottom, left:right, self.token_idx].clone() * (self.noise_scale * noise_weight.timesteps[self.cur_step])
        
        w = tmp.shape[2]
        h = tmp.shape[1]
        x = torch.linspace(0, h, h)
        y = torch.linspace(0, w, w)
        x, y = torch.meshgrid(x, y, indexing="ij")
        noise_g = dd_utils.gaussian_2d(
            x,
            y,
            mx=int(h / 2),
            my=int(w / 2),
            sx=float(h) / 2.0,
            sy=float(w) / 2.0,
        )
        
        
        noise = noise_g
        noise = (
            noise.unsqueeze(0)
            .unsqueeze(-1)
            .repeat(tmp.shape[0], 1, 1, tmp.shape[-1])
            .to(attn_replace.device)
        )

        attn_replace[:, top:bottom, left:right, self.token_idx] = tmp + noise
        mask = torch.ones_like(attn_replace, dtype=torch.bool)
        mask[:, :, right:, self.token_idx] = False
        mask[:, :, :left, self.token_idx] = False
        mask[:, :top, :, self.token_idx] = False
        mask[:, bottom:, :, self.token_idx] = False
        global_mask[..., self.token_idx] |= mask[..., self.token_idx]

        mask = torch.zeros_like(attn_replace, dtype=torch.bool)
        mask[:, top:bottom, :, self.token_idx] = True
        mask[:, :, left:right, self.token_idx] = True
        global_mask[..., self.token_idx] &= mask[..., self.token_idx]
        
        zeros_indices = torch.where(global_mask == False)
        global_mask = global_mask.clone().detach()
        
        
        if self.annealing_threshold > 0:
            global_mask |= positive_mask
            zeros_indices = torch.where(global_mask == False)
            global_mask[zeros_indices] = self.annealing_coef
            
            
        else:
            global_mask[zeros_indices] = self.annealing_coef
        
        global_mask = global_mask.half()
        attn_replace *= global_mask
        attn_replace = attn_replace.view(8, dim * dim, 77)
            
        return attn_replace, self.token_idx

    
    def set_trailing_tokens(self, tokens):
        # 49407 EOS, 49406 BOS
        self.token_idx = self.prompt_token_idx
        ids = tokens["input_ids"][0]
        for length in range(len(ids)):
            if ids[length] == 49407:
                break
        
        self.token_length = length # plot the attention maps
        
        for i in range(length, length+self.num_trailing_maps):
            if i > 76:
                break
            self.token_idx.append(i) # add the trailing tokens


def make_controller(prompts: List[str], cross_editing_steps: Dict[str, float], self_editing_steps: int, attn_kwargs: Dict, debug_kwargs: Dict) -> AttentionControlEdit:
    controller = DirectedAttention(prompts, cross_editing_steps=cross_editing_steps, self_editing_steps=self_editing_steps, attn_kwargs=attn_kwargs, debug_kwargs=debug_kwargs)
    return controller


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, edited: bool):
    out = []
    if edited:
        attention_maps = attention_store.get_average_edited_attention() # attention control edit
    else:
        attention_maps = attention_store.get_average_attention()
    
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0, save_dir="./outputs", edited=False, tags=""):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select, edited=edited)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = dd_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
        
    if edited:
        dd_utils.save_images(images=np.stack(images, axis=0), tags=f"attention_visualization_{tags}", save_dir=save_dir)
    else:
        dd_utils.save_images(images=np.stack(images, axis=0), tags=f"attention_visualization", save_dir=save_dir)
    

############################## null text inversion ##################################
class NullInversion:
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        for i in tqdm(range(NUM_DDIM_STEPS)):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                pass
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        return uncond_embeddings_list
    
    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        dd_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings
        
    
    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        
        

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((WIDTH, HEIGHT)))
    return image


################################# Directed Diffusion generation ###############################

@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    use_attn_edit: bool=True,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image',
    region=None,
):
    batch_size = len(prompt)
    dd_utils.register_attention_control(model, controller)
    height = HEIGHT
    width = WIDTH
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    
    if use_attn_edit:
        controller.set_trailing_tokens(text_input)
    
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = dd_utils.init_latent(latent, model, height, width, generator, batch_size)
    
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        
        latents = dd_utils.diffusion_step(model, latents, context, t, guidance_scale, low_resource=False)
        
        if LOG_DIFFUSION and latents.shape[0] > 1:
            image = dd_utils.latent2image(model.vae, latents)
            dd_utils.save_images(images=[image[1]], save_dir="diffusion_process", tags=str(t), region=region)
            
    if return_type == 'image':
        image = dd_utils.latent2image(model.vae, latents)
    else:
        image = latents
        
    return image, latent


def run_and_display(prompts, controller, use_attn_edit=True, latent=None, generator=None, uncond_embeddings=None, verbose=True, save_dir="./outputs", tags="", region=None, plot_region=False):
    images, x_t = text2image_ldm_stable(ldm_stable, 
                                        prompts, 
                                        controller,
                                        use_attn_edit=use_attn_edit, 
                                        latent=latent, 
                                        num_inference_steps=NUM_DDIM_STEPS, 
                                        guidance_scale=GUIDANCE_SCALE, 
                                        generator=generator, 
                                        uncond_embeddings=uncond_embeddings,
                                        region=region)
    if verbose:
        if images.shape[0] == 1:
            dd_utils.save_images(images=[images[0]], save_dir=save_dir, tags=tags, region=region, plot_region=plot_region)
        else:
            dd_utils.save_images(images=[images[1]], save_dir=save_dir, tags=tags, region=region, plot_region=plot_region)  
            
    return images, x_t


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument(
        "--image_path",
        type=str,
        default="./example_images/gnochi_mirror.jpeg",
        help="Path to the image that you want to edit.",
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="the prompt description of the image that you want to edit.",
    )
    parser.add_argument(
        "--editing_steps",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 15],
        help="the number of cross attention map editing steps",
    )
    
    parser.add_argument(
        "--self_editing_steps",
        type=int,
        default=0,
        help="the number of self attention map editing steps",
    )
    
    parser.add_argument(
        "--region",
        type=float,
        nargs="+",
        default=[],
        help="the bounding box that you want your edited object to appear.",
    ),
    
    parser.add_argument(
        "--num_trailing_maps",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20],
        help="the number of trailing maps to edit"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="output image file name"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="the directory to save the generated images"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="the path to save the generated images"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="preload model path"
    )
    parser.add_argument(
        "--prompt_indice",
        type=int,
        nargs="+",
        default=[],
        help="the "
    )
    
    parser.add_argument(
        "--noise_level",
        type=float,
        default=0.5,
        help="The gaussian noise level"
    )
    
    parser.add_argument(
        "--annealing_coef",
        type=float,
        default=0.01,
        help="the attention annealing coefficient"
    )
    
    parser.add_argument(
        "--max_attn_size",
        type=int,
        default=8192,
        help="the max size of attention maps that we edit"
    )
    
    parser.add_argument(
        "--annealing_threshold",
        type=float,
        default=0,
        help="the part that we anneal in a attention map"
    )
    
    parser.add_argument(
        "--debug",
        action='store_true',
        help="The debug version will print out the attention maps of each token at every diffusion step"
    )
    
    parser.add_argument(
        "--use_base_edit",
        type=bool,
        default=False,
        help="Whether to do edit based on the original diffusion process, or on the edited diffusion process"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="which device to use: cuda:0, cuda:1, cpu"
    )
    
    parser.add_argument(
        "--plot_region",
        action="store_true",
    )
    
    parser.add_argument(
        "--start_from",
        type=int,
        default=0,
        help="the path to save the generated images"
    )
    
    parser.add_argument(
        "--end_with",
        type=int,
        default=1000,
        help="the path to save the generated images"
    )
    
    
    args = parser.parse_args()
    args.output_path = os.path.join(args.output_dir, args.experiment_name)
    args.debug_path = os.path.join("logs", args.prompt.replace(" ", "_"), f"region_{args.region}", args.experiment_name)
    
    os.makedirs(args.output_path, exist_ok=True)
    
    print(args)
    return args



if __name__ == "__main__":
    
    args = parse_args()
    prompt = args.prompt
    
    device = args.device
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.model_path, use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
    tokenizer = ldm_stable.tokenizer

    
    image_path = args.image_path
        
    ################################
    ##  Inversion  
    ################################    
        
    null_inversion = NullInversion(ldm_stable)
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, prompt, offsets=(0,0,0,0), verbose=True)

    # visualization of inversion
    generator = torch.Generator(device=device).manual_seed(0)
    prompts = [prompt]

    
    ######################################
    ##  Directed Diffusion + prompt to prompt
    ######################################
    prompts = [prompt, prompt]

    attn_kwargs = {
        "region": args.region,
        "token_idx": args.prompt_indice,
        "noise_scale": 0.5,
        "num_trailing_maps": args.num_trailing_maps,
        "annealing_coef": args.annealing_coef,
        "max_attn_size": args.max_attn_size,
        "annealing_threshold": args.annealing_threshold,
        "use_base_edit": args.use_base_edit
    }

    trailing_maps_list = args.num_trailing_maps
    editing_steps_list = args.editing_steps


    for num_trailing_maps in trailing_maps_list:
        for editing_steps in editing_steps_list:
            
            noise_weight.set_timesteps(editing_steps)
            attn_kwargs["num_trailing_maps"] = num_trailing_maps
            cross_editing_steps = editing_steps
            
            print("num_trailing_maps: ", num_trailing_maps, "editing_steps: ", editing_steps)
            
            debug_kwargs = {}
            if DEBUG:
                debug_kwargs["output_path"] = args.debug_path
                debug_kwargs["tags"] = f"editing_{cross_editing_steps}_trailing_{num_trailing_maps}"
                
            controller = make_controller(prompts, cross_editing_steps, args.self_editing_steps, attn_kwargs, debug_kwargs=debug_kwargs)
            images, _ = run_and_display(prompts, 
                                        controller,
                                        generator=generator, 
                                        latent=x_t, 
                                        uncond_embeddings=uncond_embeddings, 
                                        save_dir=args.output_path,
                                        tags=f"editing_{cross_editing_steps}_trailing_{num_trailing_maps}",
                                        region=args.region,
                                        plot_region=args.plot_region)    