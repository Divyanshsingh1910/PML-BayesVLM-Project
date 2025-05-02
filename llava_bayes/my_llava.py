# custom llava forward pass
from typing import List, Optional, Tuple, Union
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, LlavaConfig # Assuming LlavaConfig is needed by your custom model
from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration, LlavaCausalLMOutputWithPast
from transformers.models.llava.configuration_llava import LlavaConfig

from transformers.utils import (
    # add_start_docstrings,
    # add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    # logging,
    # replace_return_docstrings,
)

import os, json, sys 
sys.path.append(".")
sys.path.append("..")
from bayesvlm.hessians import load_hessians, my_compute_covariances, optimize_prior_precision

VERBOSE = os.environ.get("verbose")

def print_(str):
    if VERBOSE:
        print_(str)
    
class CustomLlavaModel(LlavaForConditionalGeneration):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        self.input_ids = None
        self.pseudo_data_count = 10
        self.hessian_dir = "hessians/hessian_CLIP-ViT-L-14-laion2B-s32B-b82K"
        # self.device = "cuda:0"
        
    def end_here(self):
        print_("Exiting from the forward pass of the model")
        print_(f"input_ids shape: {self.input_ids.shape}")
        exit(0)
        return self.input_ids
        
    def get_img_cov_matrix(self, image_activation):

        A_img, B_img = load_hessians(self.hessian_dir, tag='img', return_info=False)
        print_(f"Loaded hessians: A_img: f{A_img.shape}\t B_img: f{B_img.shape}")

        print_("[1] Optimizing prior precision...")
        info = {
            'n_img': self.pseudo_data_count,
            'n_txt': self.pseudo_data_count,
        }
        # if prior_precision_info.json file exists then just load the values
        # else optimize the prior precision
        if os.path.exists(os.path.join(self.hessian_dir, 'prior_precision_analytic.json')):
            with open(os.path.join(self.hessian_dir, 'prior_precision_analytic.json')) as f:
                temp = json.load(f)
                info['lambda_img'] = temp['lambda_img']
                info['lambda_txt'] = temp['lambda_txt']
        else:
            info['lambda_img'] = optimize_prior_precision(
                self.multi_modal_projector,
                A=A_img,
                B=B_img,
                lmbda_init=300,
                n=info['n_img'],
                lr=1e-2,
                num_steps=1000,
                # device=self.device,
                verbose=False,
            ).item()

        print_(f"\tn_img: {info['n_img']}")
        print_(f"\tlambda_img: {info['lambda_img']}")

        with torch.no_grad():
            image_activation = image_activation.unsqueeze(0)

            cov_info_image = my_compute_covariances(A_img, B_img, info)
            A_img_inv = cov_info_image.A_inv
            A_img_inv = A_img_inv.to("cuda:0").bfloat16()
            # B_img_inv = cov_info_image.B_inv.diagonal()
            # B_img_inv = B_img_inv.to("cuda:0").bfloat16()
            
            # hack
            print_(f"A_img_inv: f{A_img_inv.shape}")
            print_(f"image activations: f{image_activation.shape}")
            # cov_image = torch.einsum('ij,jk,ik->i', image_activation, A_img_inv, image_activation)[:,None] * B_img_inv
            input_variance_proj = torch.einsum('bpi,ij,bpj->bp',
                                        image_activation,
                                        A_img_inv,
                                        image_activation)
            diag_B_inv = torch.ones(4096, device=input_variance_proj.device, dtype=input_variance_proj.dtype)
            # input_variance_proj shape: [batch, n_patches] (e.g., [1, 576])

            cov_image = input_variance_proj.unsqueeze(-1) * diag_B_inv 
        
        print_(f"Covariance: {cov_image.shape}")
        return cov_image

    @torch.no_grad()
    def my_get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: Optional[str] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        This should run the BayesVLM and return a sampled image feature of the input image.
        The projection layers should be from the Llava model since they have trained the 
        weights of these projection layers while keeping the rest image encoder frozen.
        """
        print_("###[LOG]: Entered the `my_get_image_features` function")
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")

        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True, **kwargs)

        # If we have one vision feature layer, return the corresponding hidden states,
        # otherwise, select the hidden states of each feature layer and concatenate them
        if isinstance(vision_feature_layer, int):
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            if vision_feature_select_strategy == "default":
                selected_image_feature = selected_image_feature[:, 1:]
        else:
            hs_pool = [image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
            # For default; crop CLS from each hidden state in the hidden state pool
            if vision_feature_select_strategy == "default":
                hs_pool = [hs[:, 1:] for hs in hs_pool]
            selected_image_feature = torch.cat(hs_pool, dim=-1)

        # we change this apply our own projection layer
        print_(f"selected_image_feature: {type(selected_image_feature)}, {selected_image_feature.shape}") 
        image_features = self.multi_modal_projector(selected_image_feature) #embedding
        print_(f"projected image_features: {type(image_features)}, {image_features.shape}") 

        Hack_Llava = True
        beta = 0.9
        if Hack_Llava:
            mean_img_embed = image_features.squeeze(0) #\mu 
            print_(f"mean image emb: {mean_img_embed.shape}")
            img_cov_matrix = self.get_img_cov_matrix(selected_image_feature.squeeze(0)) #\Sigma 

            # embed_distribution = torch.distributions.MultivariateNormal(mean_img_embed, img_cov_matrix)

            # sampled_img_embed = embed_distribution.sample() #\tilde{x}
            # sampled_img_embed = sampled_img_embed.unsqueeze(0)
            std_dev = img_cov_matrix.sqrt()

            distribution = torch.distributions.Normal(mean_img_embed, std_dev)
            
            sampled_img_embed = distribution.rsample()

            print_(f"Sampled image embedding: f{sampled_img_embed.shape}")

            # balanced any noisy embedding
            sampled_img_embed = (1-beta) * mean_img_embed + beta * sampled_img_embed
            return sampled_img_embed

        print_("###[LOG]: Leaving the `my_get_image_features` function")
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: torch.Tensor = None,
        **lm_kwargs,
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:

        self.input_ids = input_ids
        print_("###[LOG]: Entering the `forward` function")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if pixel_values is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # this is where the image_features will sampled
            image_features = self.my_get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
            )
            # print_(f"image_features: {type(image_features)}, {image_features.shape}") # Tensor, torch.Size([1, 576, 4096])
            # print_(f"pixel_values: {type(pixel_values)}, {pixel_values.shape}") # Tensor, torch.Size([1, 3, 336, 336])
            # print_(f"image_sizes: {type(image_sizes)}, {image_sizes}")

            special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = (input_ids == self.config.image_token_index).sum()
                n_image_features = image_features.shape[0] * image_features.shape[1]
                print_(f"Image --> tokens: {n_image_tokens}, features {n_image_features}")
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            print_(f"inputs_embeds: {type(inputs_embeds)}, {inputs_embeds.shape}") # <-- this has both image+prompt

        print_("pixel_values is None: --> should happen couple of times!")
        print_(f"inputs_embeds: {type(inputs_embeds)}, {inputs_embeds.shape}")
        # print_(type(position_ids))
        print_(f"position_ids: {position_ids}") # this keeps +=1 on generation of each token
        # print_(type(past_key_values))
        # print_(past_key_values)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        # print_(f"outputs from the LM: {type(outputs)}, {outputs}")

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -(logits.shape[1] - 1) :].to(logits.device)
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            print_("###[LOG]: Leaving the `forward` function -> not return_dict")
            return (loss,) + output if loss is not None else output

        print_("###[LOG]: Leaving the `forward` function -> CausalLMOutputwithPast")
        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )

