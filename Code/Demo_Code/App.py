import os
import streamlit as st
from PIL import Image
import torch
import torchvision

# --- Config Environment ---
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# --- Import Model 1: EfficientNet ---
from models.EfficientNetV2_Transformer.E_T_image_caption_model import Imagecaptionmodel, position_encoding
from utils.E_T_utils.image_utils import extract_image_feature
from utils.E_T_utils.model_utils import load_model_and_vocabulary
from utils.E_T_utils.caption_utils import generate_caption

# --- Import Model 2: CLIPCap 1-Stage (MLP) ---
import clip
from transformers import AutoTokenizer
from models.CLIPCap.ClipCaptionModel import ClipCaptionPrefix
from models.CLIPCap.generate import generate_beam

# --- Import Model 3: CLIPCap 2-Stage (Transformer) ---
from models.CLIPCAP_2stages.clipcap_2stage_script import CLIPCap2StagePredictor

# --- Import Metrics ---
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge

st.set_page_config(page_title="Image Captioning System", layout="wide")
session_state = st.session_state
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------------------------------------------
# 1. LOAD MODEL 1: EfficientNet + Transformer
# ----------------------------------------------------------------
@st.cache_resource
def load_ET_resources():
    try:
        E_T_MODEL_PATH = './models/EfficientNetV2_Transformer/Bestmodel.pth'
        E_T_VOCAB_PATH = './models/EfficientNetV2_Transformer/vocabulary_data.pkl'
        model, word_to_idx, idx_to_word, start_token, pad_token = load_model_and_vocabulary(E_T_MODEL_PATH, E_T_VOCAB_PATH)
        
        eff_net = torchvision.models.efficientnet_v2_s(weights="DEFAULT").to(device)
        eff_net.eval()
        eff_layer = eff_net._modules.get('features')
        return model, word_to_idx, idx_to_word, start_token, pad_token, eff_layer
    except Exception as e:
        print(f"Error loading ET: {e}")
        return None

# ----------------------------------------------------------------
# 2. LOAD MODEL 2: CLIPCap 1-Stage (MLP)
# ----------------------------------------------------------------
@st.cache_resource
def load_CLIPCap_1stage_resources():
    try:
        prefix_length = 10
        clip_model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
        tokenizer = AutoTokenizer.from_pretrained("imthanhlv/gpt2news")
        
        model = ClipCaptionPrefix(prefix_length)
        path = "models/CLIPCap/CLIPCAP_1_stage"
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model = model.eval().to(device)
        return clip_model, preprocess, tokenizer, model
    except Exception as e:
        print(f"Error loading 1-Stage: {e}")
        return None

# ----------------------------------------------------------------
# 3. LOAD MODEL 3: CLIPCap 2-Stage (Transformer)
# ----------------------------------------------------------------
@st.cache_resource
def load_CLIPCap_2stage_predictor():
    try:
        path = "models/CLIPCAP_2stages/2stage"
        predictor = CLIPCap2StagePredictor(model_path=path, device=device, prefix_length=10)
        return predictor
    except Exception as e:
        print(f"Error loading 2-Stage: {e}")
        return None

# --- INIT MODELS ---
with st.spinner("Đang tải các models..."):
    et_res = load_ET_resources()
    if et_res:
        E_T_model, word_to_idx, idx_to_word, start_token, pad_token, model_img_layer_4 = et_res
    
    clip1_res = load_CLIPCap_1stage_resources()
    if clip1_res:
        clip_model_1s, preprocess_1s, tokenizer_1s, model_1s = clip1_res
        
    predictor_2s = load_CLIPCap_2stage_predictor()

max_sequence_len = 35


def run_clipcap_1stage_inference(image_file):
    image = Image.open(image_file).convert("RGB")
    image_tensor = preprocess_1s(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model_1s.encode_image(image_tensor).to(device, dtype=torch.float32)
        prefix_embed = model_1s.clip_project(prefix).reshape(1, 10, -1)
        generated_text = generate_beam(model_1s, tokenizer_1s, embed=prefix_embed, temperature=1, beam_size=15)[0]
    return generated_text

def compute_scores(hypotheses, ground_truths):
    if not all(isinstance(g, list) for g in ground_truths):
        ground_truths = [[g] if isinstance(g, str) else g for g in ground_truths]
    
    gts = {i: ground_truths[i] for i in range(len(ground_truths))}
    res = {i: [hypotheses[i]] for i in range(len(hypotheses))}
    
    bleu_scorer = Bleu(n=4)
    rouge_scorer = Rouge()
    
    bleu, _ = bleu_scorer.compute_score(gts, res)
    rouge, _ = rouge_scorer.compute_score(gts, res)
    
    return {
        "BLEU-1": bleu[0], "BLEU-4": bleu[3], "ROUGE-L": rouge
    }


menu = st.sidebar.radio("Menu", ["Tạo Caption", "Lịch sử Generate"])

if menu == "Tạo Caption":
    st.title('📸 Image Captioning (3 Models)')
    
    col1, col2 = st.columns([1, 1])
    with col1:
        image_file = st.file_uploader("Upload ảnh", type=["jpg", "jpeg", "png", "webp"])
        if image_file:
            st.image(image_file, caption="Ảnh Input")

    with col2:
        if image_file:
            image_mode = st.radio("Chế độ:", ["Ảnh không có ground truth", "Ảnh có ground truth"])
            ground_truths = ""
            if image_mode == "Ảnh có ground truth":
                ground_truths = st.text_area("Nhập Ground Truth (mỗi dòng 1 câu):", height=120)
            
            all_models = ['EfficientNet v2 + Transformer', 'CLIPCap 1-Stage', 'CLIPCap 2-Stage']
            selected_models = st.multiselect('Chọn Model:', all_models, default=all_models[0])
            btn_run = st.button("🚀 Tạo Caption", type="primary")

    if image_file and 'btn_run' in locals() and btn_run:
        if not selected_models:
            st.warning("Chọn ít nhất 1 model.")
        else:
            captions = {}
            st.divider()

            # --- MODEL 1: EFFICIENTNET ---
            if 'EfficientNet v2 + Transformer' in selected_models and et_res:
                with st.spinner("Running EfficientNet..."):
                    image_file.seek(0)
                    img_emb = extract_image_feature(model_img_layer_4, image_file, device)
                    cap = generate_caption(E_T_model, word_to_idx, idx_to_word, img_emb, max_sequence_len, start_token, 15, pad_token, device)
                    captions['EfficientNet v2 + Transformer'] = cap
                    st.success(f"**EfficientNet:** {cap}")

            # --- MODEL 2: CLIPCAP 1-STAGE ---
            if 'CLIPCap 1-Stage' in selected_models and clip1_res:
                with st.spinner("Running CLIPCap 1-Stage..."):
                    image_file.seek(0)
                    cap = run_clipcap_1stage_inference(image_file)
                    captions['CLIPCap 1-Stage'] = cap
                    st.info(f"**CLIPCap 1-Stage:** {cap}")

            # --- MODEL 3: CLIPCAP 2-STAGE ---
            if 'CLIPCap 2-Stage' in selected_models and predictor_2s:
                with st.spinner("Running CLIPCap 2-Stage..."):
                    image_file.seek(0)
                    cap = predictor_2s.generate(image_file)
                    captions['CLIPCap 2-Stage'] = cap
                    st.warning(f"**CLIPCap 2-Stage:** {cap}")

            # --- SCORE ---
            scores_result = {}
            if image_mode == "Ảnh có ground truth" and ground_truths.strip():
                gt_list = ground_truths.split("\n")
                st.subheader("📊 Kết quả Metrics")
                cols = st.columns(len(captions))
                for idx, (m_name, cap) in enumerate(captions.items()):
                    s = compute_scores([cap], [gt_list])
                    scores_result[m_name] = s
                    with cols[idx]:
                        st.write(f"**{m_name}**")
                        st.json(s)

            # --- HISTORY ---
            if 'history' not in session_state: session_state.history = []
            session_state.history.append({
                "image": image_file,
                "captions": captions,
                "ground_truths": ground_truths if image_mode == "Ảnh có ground truth" else None,
                "scores": scores_result
            })

elif menu == "Lịch sử Generate":
    st.title("Lịch sử")
    if 'history' in session_state and session_state.history:
        for idx, r in enumerate(reversed(session_state.history)):
            st.markdown(f"### #{len(session_state.history)-idx}")
            st.image(r['image'], width=300)
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Captions:**")
                for m, c in r['captions'].items(): st.write(f"- **{m}**: {c}")
                if r['ground_truths']: st.info(f"GT: {r['ground_truths']}")
            with c2:
                if r['scores']:
                    for m, s in r['scores'].items():
                        with st.expander(f"Score {m}"): st.json(s)
            st.divider()
    else:
        st.info("Chưa có lịch sử.")
