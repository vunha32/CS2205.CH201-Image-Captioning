import pickle
import torch
from models.EfficientNetV2_Transformer.E_T_image_caption_model import Imagecaptionmodel

# @st.cache_resource
def load_model_and_vocabulary(model_path, vocab_path):
    with open(vocab_path, 'rb') as f:
        vocabulary_data = pickle.load(f)
    
    word_to_idx = vocabulary_data['word_to_idx']
    idx_to_word = vocabulary_data['idx_to_word']
    start_token = word_to_idx['<start>']
    pad_token = word_to_idx['<pad>']
    
    model = Imagecaptionmodel(vocab_size=1823)
    # Thêm weights_only=False để cho phép load model cũ
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()

    return model, word_to_idx, idx_to_word, start_token, pad_token
