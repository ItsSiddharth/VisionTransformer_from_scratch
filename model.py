import torch
import torch.nn as nn
import torch.nn.init as init
from transformers.modeling_outputs import BaseModelOutput

import math

class layerNorm(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # norm = (x-mean(x))/sqrt(std(x)+eps)*alpha + beta
        numerator = x - x.mean(dim = -1, keepdim=True)
        denominator = math.sqrt((x.std(dim = -1, keepdim=True) + self.epsilon))
        return (numerator/denominator) * self.alpha + self.beta

class PatchEmbeddings(nn.Module):
    def __init__(self, image_size: int= (224, 224), patch_size: int = (16, 16), num_channels: int = 3, emb_dims: int = 768) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.emb_dims = emb_dims
        self.num_patches = (image_size[0]//patch_size[0]) * (image_size[1]//patch_size[1])

        self.projection = nn.Conv2d(self.num_channels, self.emb_dims, self.patch_size, self.patch_size)

    def forward(self, input_image): #(batch, channel, height, width)
        batch_size, num_channels, height, width = input_image.shape
        # we have to check if the image is properly resized
        assert height*width == self.image_size[0]*self.image_size[1], "Resize image to correct dimensions before parsing"

        # (b, c, h, w) --> (b, emb_dim, h ,w) --> (b, emb_dim, h*w) --> (b, h*w, emb_dim)
        return self.projection(input_image).flatten(2).transpose(1, 2)

class ViTinputEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["hidden_dims"])) # (batch, the_token_itself, same_dim_as_image_patch_emb)
        self.patch_embeddings = PatchEmbeddings(
            image_size=(config["image_size"], config["image_size"]),
            patch_size=(config["patch_size"], config["patch_size"]),
            num_channels=config["num_channels"],
            emb_dims=config["hidden_dims"]
        )

        num_patches = self.patch_embeddings.num_patches
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, config["hidden_dims"])) # positional Embb
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, input_image):
        batch_size = input_image.shape[0]
        image_embedding = self.patch_embeddings(input_image) # (b, h*w, emb_dim)
        
        cls_token = self.cls_token.expand(batch_size, -1, -1) # (1, 1, emb_dim) --> (batch_size, 1, emb_dim)
        cls_and_image_emb = torch.cat((cls_token, image_embedding), dim=1) # dim!=0 because that is the batch dimension
        final_embedding = cls_and_image_emb + self.positional_embedding
        final_embedding_with_dropout = self.dropout(final_embedding)

        return final_embedding_with_dropout
    
class ViTSelfAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config["hidden_dims"]
        self.num_attn_heads = config["num_attn_heads"]
        assert self.hidden_dim % self.num_attn_heads == 0, "The hidden_dims is not divisible by num_attn_heads"
        self.attention_head_size = self.hidden_dim//self.num_attn_heads
        self.attention_map_size = self.attention_head_size * self.num_attn_heads

        self.query = nn.Linear(self.hidden_dim, self.attention_map_size)
        self.key = nn.Linear(self.hidden_dim, self.attention_map_size)
        self.value = nn.Linear(self.hidden_dim, self.attention_map_size)

        self.dropout = nn.Dropout(config["dropout"])

    def convert_linear_map_to_multihead(self, linear_map):
        # linear_map_shape = (batch, seq_len, hidden_dim) --> (batch, seq_len, num_attn_head, attn_head_size)
        multi_head_shape = linear_map.size()[:-1] + (self.num_attn_heads, self.attention_head_size)
        multi_head_linear_map = linear_map.view(*multi_head_shape)

        return multi_head_linear_map.permute(0, 2, 1, 3) # (batch, num_attn_head, seq_len, attn_head_size)
    
    def forward(self, input_embedding, return_score=False):
        query = self.query(input_embedding)
        key = self.key(input_embedding)
        value = self.value(input_embedding)

        query_multihead = self.convert_linear_map_to_multihead(query)
        key_multihead = self.convert_linear_map_to_multihead(key)
        value_multihead = self.convert_linear_map_to_multihead(value)

        attention_score = torch.matmul(query_multihead, key_multihead.transpose(-1, -2))/math.sqrt(self.attention_head_size)
        attention_probabilities = nn.Softmax(dim=-1)(attention_score)

        attention_dropout = self.dropout(attention_probabilities)

        final_attn_output = torch.matmul(attention_dropout, value_multihead)
        final_attn_output = final_attn_output.permute(0, 2, 1, 3).contiguous() # (batch, seq_len, num_attn_head, attn_head_size)
        reshape_for_final_attn = final_attn_output.size()[:-2] + (self.attention_map_size, )
        reshaped_final_attn_layer = final_attn_output.view(*reshape_for_final_attn)

        return (reshaped_final_attn_layer, attention_probabilities) if return_score else (reshaped_final_attn_layer,)

class ViTBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.multi_head_attn = ViTSelfAttention(config)
        self.layerNorm1 = layerNorm()
        self.layerNorm2 = layerNorm()
        self.mlp_dense_layer1 = nn.Linear(config["hidden_dims"], config["upsample_mlp_dims"])
        self.activation_fnc = nn.GELU()
        self.mlp_dense_layer2 = nn.Linear(config["upsample_mlp_dims"], config["hidden_dims"])
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, image_embedding,return_score=False):
        normalized_embedding = self.layerNorm1(image_embedding)
        attention_block_outputs = self.multi_head_attn(normalized_embedding, return_score)

        attention_output_layer = attention_block_outputs[0]
        attention_probabilities = attention_block_outputs[1:]

        residual_connection_1 = image_embedding + attention_output_layer
        residual_normalization = self.layerNorm2(residual_connection_1)

        mlp_upsample_output = self.mlp_dense_layer1(residual_normalization)
        mlp_upsample_activ = self.activation_fnc(mlp_upsample_output)

        mlp_downsample_output = self.mlp_dense_layer2(mlp_upsample_activ)
        mlp_dropout = self.dropout(mlp_downsample_output)

        residual_connection_2 = mlp_dropout + residual_connection_1

        final_output = (residual_connection_2, ) + attention_probabilities

        return final_output

class ViTEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ViTBlock(config) for _ in range(config["num_attn_blocks"])])

    def forward(self, input_embedding, return_scores=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        all_attention_scores = () if return_scores else None

        curr_hidden_state = input_embedding
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (curr_hidden_state, )
            
            layer_outputs = layer(curr_hidden_state, return_scores)
            curr_hidden_state = layer_outputs[0]

            if return_scores:
                all_attention_scores = all_attention_scores + (layer_outputs[1], )
            
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (curr_hidden_state, )
        
        return tuple(v for v in [curr_hidden_state, all_hidden_states, all_attention_scores] if v is not None)
    
class ViTPooler(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config["hidden_dims"], config["hidden_dims"])
        self.activation = nn.Tanh()

    def forward(self, hidden_state):
        cls_embedding = hidden_state[:, 0]
        pooled_output = self.dense(cls_embedding)
        pooled_activation = self.activation(pooled_output)
        return pooled_activation

class ViTModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.embedding = ViTinputEmbedding(config)
        self.encoder = ViTEncoder(config)

        self.layerNorm = layerNorm()
        self.pooler = ViTPooler(config)
    
    def forward(self, input_image, return_scores=False, output_hidden_states=False):
        image_embedding = self.embedding(input_image)
        encoder_output = self.encoder(image_embedding, return_scores, output_hidden_states)
        attention_output = encoder_output[0]
        attention_output_norm = self.layerNorm(attention_output)
        pooled_output = self.pooler(attention_output_norm)

        return (attention_output_norm, pooled_output) + encoder_output[1:]
        

class ViTforImageClassification(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.num_labels = config["num_labels"]

        self.vit_model = ViTModel(config)
        self.classifier = nn.Linear(config["hidden_dims"], self.num_labels)

    def forward(self, image_input, labels=None, return_scores=False, output_hidden_states=False):
        outputs = self.vit_model(image_input, return_scores, output_hidden_states)

        attention_output = outputs[0]
        classifier_output = self.classifier(attention_output[:, 0, :])

        loss = None
        if labels is not None:
            assert self.num_labels != 1, "Regression not implemented"
            if self.num_labels > 1:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(classifier_output.view(-1, self.num_labels), labels.view(-1))
        
        final_output = (classifier_output, ) + outputs[2:] # outputs = (attention_output_norm, pooled_output) + encoder_output[1:]

        return ((loss, ) + final_output) if labels is not None else final_output

                
