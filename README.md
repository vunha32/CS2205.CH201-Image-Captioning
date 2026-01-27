# ViClipCap: Tinh chỉnh Tiền tố CLIP cho Bài toán Sinh mô tả ảnh Tiếng Việt

> **Đồ án môn học:** CS2205 - Phương pháp Nghiên cứu Khoa học trong Công nghệ Thông tin  
> **Sinh viên:** Nguyễn Hà Anh Vũ  
> **MSSV:** 250101077  
> **Lớp:** CS2205.CH201

---

## 📖 Tóm tắt (Abstract)

Dự án này giới thiệu **ViClipCap**, một phương pháp hiệu quả để sinh mô tả ảnh tiếng Việt (Vietnamese Image Captioning) bằng cách tận dụng kiến trúc **Prefix Tuning**. Thay vì huấn luyện lại từ đầu các mô hình ngôn ngữ lớn, chúng tôi đề xuất chiến lược "đứng trên vai người khổng lồ": kết hợp khả năng trích xuất đặc trưng thị giác của **CLIP (ViT-B/32)** và khả năng sinh văn bản của **Vietnamese GPT-2**.

Hệ thống giữ nguyên trọng số (Frozen) của hai mô hình nền tảng và chỉ huấn luyện một **Mạng Ánh xạ (Mapping Network)** siêu nhẹ. Thực nghiệm trên bộ dữ liệu **KTVIC (Life Domain)** cho thấy mô hình đạt hiệu suất vượt trội so với baseline (CNN+LSTM) với chi phí tính toán tối thiểu.

---

## 🚀 Kiến trúc Mô hình (Methodology)

Hệ thống ViClipCap bao gồm 3 thành phần chính:

1.  **Encoder (Frozen):** Sử dụng `CLIP ViT-B/32` để trích xuất *Visual Embeddings* từ ảnh đầu vào.
2.  **Mapping Network (Trainable):** Một mạng MLP/Transformer nhẹ đóng vai trò cầu nối, chuyển đổi đặc trưng ảnh thành chuỗi *Prefix Embeddings* (Soft Prompts).
3.  **Decoder (Frozen):** Mô hình `Vietnamese GPT-2` nhận chuỗi Prefix và sinh ra câu mô tả tiếng Việt tự nhiên.

![Architecture](./images/architecture.jpg)

### Điểm nổi bật kỹ thuật:
* **Lightweight:** Chỉ cập nhật tham số $\theta$ của Mapping Network.
* **Prevention of Catastrophic Forgetting:** Không làm mất tri thức ngôn ngữ đã học của GPT-2.
* **Efficiency:** Tốc độ suy diễn nhanh, yêu cầu tài nguyên phần cứng thấp.

---

## 📊 Kết quả Thực nghiệm (Experimental Results)

Mô hình được huấn luyện và đánh giá trên bộ dữ liệu **KTVIC** (4.327 ảnh, 21.635 captions). Kết quả so sánh với mô hình cơ sở (Baseline CNN+LSTM) như sau:

| Metric | CNN + LSTM | **ViClipCap (Ours)** | Cải thiện |
| :--- | :---: | :---: | :---: |
| **BLEU-4** | 0.2572 | **0.3431** | 🟢 +8% |
| **ROUGE-L** | 0.4895 | **0.5204** | 🟢 +3% |
| **CIDEr** | 0.6282 | **0.8127** | 🟢 +18% |
| **METEOR** | 0.2995 | **0.3194** | 🟢 +2% |
| **SPICE** | 0.0782 | **0.0829** | 🟢 +1% |

> **Nhận xét:** ViClipCap vượt trội hoàn toàn trên mọi chỉ số, đặc biệt là CIDEr (độ tương đồng ngữ nghĩa) và BLEU-4 (độ chính xác từ vựng).

---

## 🛠️ Cài đặt & Hướng dẫn sử dụng (Installation & Usage)

### 1. Yêu cầu hệ thống
* Python 3.8+
* PyTorch 1.9+
* Transformers (Hugging Face)
* CUDA (khuyến nghị để huấn luyện nhanh hơn)

### 2. Cài đặt thư viện
```bash
git clone [https://github.com/username/CS2205.CH201-Image-Captioning.git](https://github.com/username/CS2205.CH201-Image-Captioning.git)
cd CS2205.CH201-Image-Captioning
pip install -r requirements.txt
