import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, AutoTokenizer
import clip
from PIL import Image

# ======================================================
# 1. ĐỊNH NGHĨA KIẾN TRÚC MODEL (Dành riêng cho 2-Stage)
# ======================================================

class TransformerMapper(nn.Module):
    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=8, batch_first=True),
            num_layers=num_layers
        )
        self.clip_project = nn.Linear(dim_clip, dim_embedding * clip_length)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        x = self.clip_project(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        inputs = torch.cat((x, prefix), dim=1)
        out = self.transformer(inputs)
        return out[:, self.clip_length:]

class ClipCaptionModel2Stage(nn.Module):
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens, prefix, mask=None, labels=None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
            
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int = 10, prefix_size: int = 512, num_layers: int = 8):
        super(ClipCaptionModel2Stage, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("imthanhlv/gpt2news")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        # Mapper 2-stage (Transformer)
        self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length, 1, num_layers)

class ClipCaptionPrefix2Stage(ClipCaptionModel2Stage):
    def __init__(self, prefix_length=10, prefix_size=512, num_layers=8):
        super(ClipCaptionPrefix2Stage, self).__init__(prefix_length, prefix_size, num_layers)

# ======================================================
# 2. HÀM GENERATE (BEAM SEARCH)
# ======================================================

def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1.0, stop_token: str = '.'):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)

        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float('inf')
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)], skip_special_tokens=True) for output, length in zip(output_list, seq_lengths)]
    
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts[0]

# ======================================================
# 3. CLASS WRAPPER CHO APP.PY
# ======================================================

class CLIPCap2StagePredictor:
    def __init__(self, model_path, device, prefix_length=10):
        self.device = device
        self.prefix_length = prefix_length
        
        print(f"[2-Stage] Loading CLIP...")
        self.clip_model, self.preprocess = clip.load("ViT-B/16", device=device, jit=False)
        
        print(f"[2-Stage] Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
        
        print(f"[2-Stage] Loading Model Architecture...")
        self.model = ClipCaptionPrefix2Stage(prefix_length=prefix_length, num_layers=8)
        
        print(f"[2-Stage] Loading Weights from {model_path}...")
        # Load weights (chú ý weights_only=False để tránh lỗi pickle)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Nếu checkpoint là dict chứa 'state_dict' hoặc 'model_state_dict'
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
        else:
             # Trường hợp load model full
             self.model.load_state_dict(checkpoint.state_dict(), strict=False)

        self.model = self.model.to(device)
        self.model.eval()

    def generate(self, image_file):
        try:
            image = Image.open(image_file).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prefix = self.clip_model.encode_image(image_tensor).float()
                prefix_embed = self.model.clip_project(prefix).reshape(1, self.prefix_length, -1)
                
                caption = generate_beam(
                    self.model,
                    self.tokenizer,
                    embed=prefix_embed,
                    beam_size=5,
                    entry_length=50
                )
            return caption
        except Exception as e:
            return f"Error 2-stage: {str(e)}"
