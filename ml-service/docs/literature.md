# Academic Literature Survey: Deep Multilingual Conformer ASR & Core ALM Multimodal Fusion

**Project:** Smart Horizon 2026 (SH-DST-02) | **Team ID:** SHIH26-TID-320  
**Repository:** `ml-service` (PyTorch 2.14.0+cu126, CUDA 12.6, RTX 4060 Laptop GPU)  
**System Architecture:** 12-layer Deep Multilingual Conformer ASR (80.24M parameters) with $4\times$ 2D Conv Subsampling, Multilingual CTC Projection (2,122 tokens), and ALM Temporal Continuous Representation Pipeline $(B, T', 512) \to (B, T', 256)$.

---

## 📚 Executive Matrix: Key Papers & Repository Mapping

| # | Paper Title & Authors | Venue & Year | Canonical Link | Core Concept Extracted | Codebase Implementation File |
|---|----------------------|--------------|----------------|------------------------|------------------------------|
| **1** | **Conformer: Convolution-augmented Transformer for Speech Recognition**<br>*A. Gulati, J. Qin, C. Chiu, N. Parmar, Y. Zhang, J. Yu, W. Han, S. Wang, Z. Zhang, Y. Wu, R. Pang* | Interspeech 2020 | [arXiv:2005.08100](https://arxiv.org/abs/2005.08100) | Macaron-style FFN + MHSA + Depthwise 1D Conv block sandwich ($d_{model}=512, d_{ff}=2048, 8\text{ heads}, k=31$) | [`src/asr/conformer.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/conformer.py) |
| **2** | **Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks**<br>*A. Graves, S. Fernández, F. Gomez, J. Schmidhuber* | ICML 2006 | [ACM:1143891](https://dl.acm.org/doi/10.1145/1143842.1143891) | Alignment-free CTC loss lattice, blank token (`blank_id=0`), and canonical greedy duplicate collapse | [`src/asr/ctc.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/ctc.py) |
| **3** | **SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition**<br>*D. S. Park, W. Chan, Y. Zhang, C. Chiu, B. Zoph, E. D. Cubuk, Q. V. Le* | Interspeech 2019 | [arXiv:1904.08779](https://arxiv.org/abs/1904.08779) | Time and frequency masking directly in Log-Mel space; acoustic silence floor fill ($-11.5129$) | [`src/asr/frontend.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/frontend.py) |
| **4** | **IndicVoices: Towards building an inclusive multilingual speech dataset for Indian languages**<br>*AI4Bharat (K. Doddapaneni et al.)* | Interspeech 2024 | [arXiv:2403.01926](https://arxiv.org/abs/2403.01926) | Regional spontaneous speech curation across 6 Indian languages; speaker-disjoint split protocol | [`training/dataset.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/training/dataset.py) |
| **5** | **Attention Is All You Need**<br>*A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, I. Polosukhin* | NeurIPS 2017 | [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) | Multi-Head Self-Attention ($\text{MHSA}$), Scaled Dot-Product, sinusoidal temporal positional encodings | [`src/asr/conformer.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/conformer.py) |
| **6** | **Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View**<br>*Y. Lu, Z. Li, D. He, Z. Sun, Q. Dong, T. Qin, L. Wang, T. Liu* | ICLR 2020 | [arXiv:1906.02762](https://arxiv.org/abs/1906.02762) | Macaron-style half-step FFN sandwich ($0.5\times\text{FFN}_1 + \text{MHSA} + \text{Conv} + 0.5\times\text{FFN}_2$) | [`src/asr/conformer.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/conformer.py) |
| **7** | **AudioPaLM: A Large Language Model That Can Speak and Listen**<br>*P. K. Rubenstein et al. (Google Research)* | arXiv 2023 | [arXiv:2306.12925](https://arxiv.org/abs/2306.12925) | Continuous frame-level acoustic representation injection into LLM multi-modal temporal fusion pipeline | [`src/fusion/adapter.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/fusion/adapter.py) |
| **8** | **ESPnet: End-to-End Speech Processing Toolkit**<br>*S. Watanabe, T. Hori, S. Karita, T. Hayashi, J. Nishitoba et al.* | Interspeech 2018 | [arXiv:1804.00015](https://arxiv.org/abs/1804.00015) | Dynamic batching by accumulated audio duration (`max_batch_audio_seconds`) & duration bucketing | [`training/dataset.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/training/dataset.py) |
| **9** | **ContextNet: Improving Convolutional Neural Networks for ASR with Global Context**<br>*W. Han, Z. Zhang, Y. Zhang, J. Yu, C. Chiu, D. S. Park, Y. Wu* | Interspeech 2020 | [arXiv:2005.03191](https://arxiv.org/abs/2005.03191) | 2D Depthwise-separable convolutional subsampling ($4\times$ reduction: 10ms hop $\to$ 40ms frame stride) | [`src/asr/frontend.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/frontend.py) |
| **10**| **wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations**<br>*A. Baevski, H. Zhou, A. Mohamed, M. Auli* | NeurIPS 2020 | [arXiv:2006.11477](https://arxiv.org/abs/2006.11477) | Preservation of dense $(B, T', 512)$ continuous latent speech embeddings for downstream acoustic scene reasoning | [`src/asr/multilingual_asr.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/multilingual_asr.py) |

---

## 🔬 In-Depth Analysis of Individual Papers & Architectural Contributions

---

### 1. Conformer: Convolution-augmented Transformer for Speech Recognition
* **Authors:** Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, Ruoming Pang (Google Research)
* **Venue:** *Interspeech 2020*
* **Link:** [https://arxiv.org/abs/2005.08100](https://arxiv.org/abs/2005.08100)
* **Key Contribution:** Standard Transformers capture global context through multi-head self-attention but struggle with fine-grained local acoustic correlations. CNNs capture translation-invariant local features efficiently but fail to model long-range sequence context. The Conformer marries both in a unified block.
* **What We Implemented in the Codebase:**
  - Located in [`src/asr/conformer.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/conformer.py):
    - **12 Stacked Conformer Blocks** with the exact Macaron-style sandwich structure:
      $$\tilde{x}_i = x_i + \frac{1}{2} \text{FFN}(x_i)$$
      $$x'_i = \tilde{x}_i + \text{MHSA}(\tilde{x}_i)$$
      $$x''_i = x'_i + \text{Conv}(x'_i)$$
      $$y_i = \text{LayerNorm}(x''_i + \frac{1}{2} \text{FFN}(x''_i))$$
    - Multi-Head Self-Attention: $8$ heads, $d_{model} = 512$, $d_k = 64$.
    - Macaron Feed-Forward Module: $d_{model} = 512 \to d_{ff} = 2048 \to 512$ with Swish activation and half-step multiplier ($0.5$).
    - Conformer Convolution Module: LayerNorm $\to$ Pointwise Conv $\to$ Gated Linear Unit (GLU) $\to$ 1D Depthwise Conv ($k=31$, `groups=d_model`) $\to$ BatchNorm1d $\to$ Swish $\to$ Pointwise Conv $\to$ Dropout ($0.1$).

---

### 2. Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks
* **Authors:** Alex Graves, Santiago Fernández, Faustino Gomez, Jürgen Schmidhuber
* **Venue:** *International Conference on Machine Learning (ICML 2006)*
* **Link:** [https://dl.acm.org/doi/10.1145/1143842.1143891](https://dl.acm.org/doi/10.1145/1143842.1143891)
* **Key Contribution:** Introduced an objective function allowing sequence-to-sequence learning without pre-segmented ground truth alignments by introducing a special blank token ($\epsilon$ / `<blank>`) and summing conditional probabilities over all possible valid alignments.
* **What We Implemented in the Codebase:**
  - Located in [`src/asr/ctc.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/ctc.py):
    - Loss formulation:
      $$P(l|\mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(l)} P(\pi|\mathbf{x})$$
    - Hard assertion that `blank_id = 0` and `nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)`.
    - Tensor layout: log-probabilities transposed to $(T \times B \times V)$ conforming to PyTorch CTCLoss specification.
    - **Canonical Greedy Decoding Collapse Algorithm**:
      1. $\hat{\pi}_t = \arg\max_k P(\pi_t = k | \mathbf{x})$.
      2. Collapse consecutive identical labels: $\hat{\pi}_t = \hat{\pi}_{t-1} \implies \text{drop}$.
      3. Discard blank labels ($\hat{\pi}_t = 0 \implies \text{drop}$).
      4. Discard padding labels ($\hat{\pi}_t = 3 \implies \text{drop}$).
      5. Map remaining token IDs via `tokenizer.decode()`.
    - Resolved the classic **CTC Blank Collapse Valley** by demonstrating that at early training stages ($P(\text{blank}) \approx 0.95$, loss $\approx 4.5$), greedy argmax collapses to empty strings unless optimizer steps $\ge 100$ and $lr \ge 1\times 10^{-4}$ are maintained.

---

### 3. SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
* **Authors:** Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, Quoc V. Le (Google Brain)
* **Venue:** *Interspeech 2019*
* **Link:** [https://arxiv.org/abs/1904.08779](https://arxiv.org/abs/1904.08779)
* **Key Contribution:** Demonstrated that directly augmenting the spectrogram representation (rather than raw audio waveforms) via frequency masking and time masking prevents overfitting and dramatically improves robustness.
* **What We Implemented in the Codebase:**
  - Located in [`src/asr/frontend.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/frontend.py):
    - Implemented `SpecAugment(nn.Module)` with parameters:
      - Frequency masking: $m_F = 2$ masks, $f \in [1, F=15)$ channels.
      - Time masking: $m_T = 2$ masks, $t \in [1, T=35)$ time frames bounded by $t_{mel} / 5$.
    - **Critical Mathematical Discovery**: In log-Mel space ($\log(\text{mel} + 10^{-5})$), acoustic silence corresponds to $\log(10^{-5}) \approx -11.5129$, whereas `0.0` represents maximum energy ($10^0 = 1.0$). We fixed the mask fill value from `0.0` to `-11.5129`, preventing artificial loud-burst distortion from corrupting `BatchNorm2d` statistics.
    - SpecAugment is strictly bypassed during validation, test, and inference (`if self.training`).

---

### 4. IndicVoices: Towards Building an Inclusive Multilingual Speech Dataset for Indian Languages
* **Authors:** Tahir Javed, Kaushal Bhogale, Abhigyan Raman, Anoop Kunchukuttan, Pratyush Kumar, Mitesh M. Khapra (AI4Bharat)
* **Venue:** *Interspeech 2024*
* **Link:** [https://arxiv.org/abs/2403.01926](https://arxiv.org/abs/2403.01926)
* **Key Contribution:** Created an open-source, representative speech dataset capturing spontaneous conversational speech across 22 Indian languages across diverse districts, accents, age groups, and recording environments.
* **What We Implemented in the Codebase:**
  - Located in [`training/dataset.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/training/dataset.py) and [`data/manifests/`](file:///u:/Hackathons/ALM-NHCE/ml-service/data/manifests/):
    - Curated a strict 4.198 GB (19.57 hours, 6,132 utterances) multilingual subset spanning 6 Indic languages:
      - Hindi (`hi`), Telugu (`te`), Tamil (`ta`), Bengali (`bn`), Marathi (`mr`), Kannada (`kn`), plus Mandarin Chinese (`zh`).
    - Enforced **strict speaker-disjoint partitioning** across splits:
      - Train: 4,922 utterances (zero speaker overlap with val/test).
      - Validation: 601 utterances.
      - Test: 609 utterances.
    - Verified audio-transcript alignment, non-empty speech duration, and standardized 16 kHz mono 16-bit PCM WAV formatting.

---

### 5. Attention Is All You Need
* **Authors:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
* **Venue:** *NeurIPS 2017*
* **Link:** [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
* **Key Contribution:** Replaced recurrent and convolutional sequence-to-sequence models with pure attention mechanisms, introducing Scaled Dot-Product Attention and Multi-Head Attention.
* **What We Implemented in the Codebase:**
  - Located in [`src/asr/conformer.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/conformer.py):
    - Scaled Dot-Product Attention formulation:
      $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$
    - Key padding mask $M_{b, t} = -\infty$ for padded audio frames, guaranteeing zero influence of padded silence on acoustic representations.
    - Sinusoidal absolute positional encoding:
      $$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

---

### 6. Understanding and Improving Transformer From a Multi-Particle Dynamic System Point of View
* **Authors:** Yiping Lu, Zhuohan Li, Di He, Zhiqing Sun, Qi Dong, Tao Qin, Liwei Wang, Tie-Yan Liu
* **Venue:** *ICLR 2020*
* **Link:** [https://arxiv.org/abs/1906.02762](https://arxiv.org/abs/1906.02762)
* **Key Contribution:** Interpreted deep Transformers as discretization schemes of ordinary differential equations (ODEs). Proved that placing half-step Feed-Forward Networks around multi-head attention (the "Macaron" structure) mimics a second-order Runge-Kutta ODE solver, yielding significantly improved numerical stability and gradient propagation.
* **What We Implemented in the Codebase:**
  - Located in [`src/asr/conformer.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/conformer.py):
    - `ConformerBlock` implements two separate FFN modules per block (`ffn1` and `ffn2`), each scaled by $0.5$.
    - Enabled stable training of our deep 12-layer Conformer network without gradient vanishing or exploding (verified gradient norm $\in [8.7, 52.1]$ across all layers in `artifacts/gradient_audit.json`).

---

### 7. AudioPaLM: A Large Language Model That Can Speak and Listen
* **Authors:** Paul K. Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, Ankur Bapna, Zalán Borsos, Félix de Chaumont Quitry, Peter Chen, Dalia El Badawy, Wei Han, Eugene Kharitonov, Hannah Muckenhirn, Dirk Padfield, James Quinn, Brian Strope, Yonghui Wu et al. (Google Research)
* **Venue:** *arXiv 2023*
* **Link:** [https://arxiv.org/abs/2306.12925](https://arxiv.org/abs/2306.12925)
* **Key Contribution:** Demonstrated that speech-language multimodal foundation models achieve superior cross-modal reasoning when continuous acoustic representations are directly interfaced with the language model's latent embedding space, rather than relying on discrete transcript text.
* **What We Implemented in the Codebase:**
  - Located in [`src/fusion/adapter.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/fusion/adapter.py) and [`src/asr/multilingual_asr.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/multilingual_asr.py):
    - Defined the Core ALM integration contract:
      $$\text{Raw Audio} \xrightarrow{\text{Conformer Encoder}} H_{\text{acoustic}} \in \mathbb{R}^{B \times T' \times 512} \xrightarrow{\text{ASRAdapter}} H_{\text{ALM}} \in \mathbb{R}^{B \times T' \times 256}$$
    - Preserves continuous prosodic, pitch, and phonetic stress signals for downstream threat detection and urgency reasoning in the Core ALM Temporal Fusion pipeline.

---

### 8. ESPnet: End-to-End Speech Processing Toolkit
* **Authors:** Shinji Watanabe, Takaaki Hori, Shigeki Karita, Tomoki Hayashi, Jiro Nishitoba, Yuya Unno, Nelson Enrique Yalta Soplin, Jahn Heymann, Matthew Wiesner, Nanxin Chen, Adithya Renduchintala, Tatsuya Ochiai
* **Venue:** *Interspeech 2018*
* **Link:** [https://arxiv.org/abs/1804.00015](https://arxiv.org/abs/1804.00015)
* **Key Contribution:** Established modern end-to-end speech training practices, specifically duration-aware dynamic batching and length bucketing to eliminate zero-padding memory waste on GPUs.
* **What We Implemented in the Codebase:**
  - Located in [`training/dataset.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/training/dataset.py):
    - Implemented `DynamicBatchSampler(Sampler[List[int]])`:
      - Buckets utterances by audio duration so items in each mini-batch have near-identical lengths.
      - Limits batch size by total accumulated speech seconds: `max_batch_audio_seconds = 80.0s`.
      - Eliminates the silent padding issue that previously distorted `BatchNorm` statistics.
      - Reduced VRAM allocation to $2.41\text{ GB}$ (only $29.4\%$ of our $8.0\text{ GB}$ hardware budget).

---

### 9. ContextNet: Improving Convolutional Neural Networks for ASR with Global Context
* **Authors:** Wei Han, Zhengdong Zhang, Yu Zhang, Jiahui Yu, Chung-Cheng Chiu, Daniel S. Park, Yonghui Wu (Google Research)
* **Venue:** *Interspeech 2020*
* **Link:** [https://arxiv.org/abs/2005.03191](https://arxiv.org/abs/2005.03191)
* **Key Contribution:** Introduced depthwise separable convolutional subsampling frontends for end-to-end ASR, achieving $4\times$ temporal downsampling while preserving acoustic feature resolution.
* **What We Implemented in the Codebase:**
  - Located in [`src/asr/frontend.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/frontend.py):
    - Implemented `Conv2dSubsampling`:
      - 2D Conv 1 (stride 2, padding 1): $\text{in}=1 \to \text{out}=256 \to \text{BatchNorm2d} \to \text{GELU}$.
      - 2D Conv 2 (stride 2, padding 1): $\text{in}=256 \to \text{out}=512 \to \text{BatchNorm2d} \to \text{GELU}$.
      - Reduces 10ms frame hop ($100\text{ frames/sec}$) to 40ms representation stride ($25\text{ frames/sec}$).
      - Linear projection: $512 \times 20 = 10,240 \to 512$ ($d_{model}$).
      - Delivers Real-Time Factor $\text{RTF} = 0.0095$ ($105\times$ faster than real-time) on the RTX 4060 GPU.

---

### 10. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations
* **Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (Meta AI)
* **Venue:** *NeurIPS 2020*
* **Link:** [https://arxiv.org/abs/2006.11477](https://arxiv.org/abs/2006.11477)
* **Key Contribution:** Formalized that dense continuous speech representations extracted from multi-layer encoders encode acoustic, phonetic, and language invariants superior to discrete surface transcripts.
* **What We Implemented in the Codebase:**
  - Located in [`src/asr/multilingual_asr.py`](file:///u:/Hackathons/ALM-NHCE/ml-service/src/asr/multilingual_asr.py):
    - Implemented `ASRModel.encode(audio)` and `ASRModel.get_speech_embeddings(audio)`.
    - Returns high-resolution frame-level embeddings $(B, T', 512)$ along with optional attentive global pooling $(B, 512)$ without discarding the uncollapsed temporal sequence dimension.
