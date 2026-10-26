# How SentenceTransformer `encode()` Works Internally

When using Sentence Transformers (e.g., `SentenceTransformer("thenlper/gte-large")`), calling `model.encode(chunk)` converts textual chunks into dense numerical embedding vectors using a **4-stage deep learning pipeline**.

---

## 1. Codebase Reference (`SentenceTransformer.py`)

In `sentence_transformers/SentenceTransformer.py`:
- **Line 396**: Contains type-hinting `@overload` signatures for `encode(...)`.
- **Line 522**: Actual implementation definition `def encode(...)`.

### Exact Execution Breakdown:

| Step | What Happens | Exact Lines in `SentenceTransformer.py` |
| :--- | :--- | :--- |
| **Batching & Sorting** | Chunks are sorted by length to minimize wasted padding computation. | **Lines 648–651** |
| **1. Tokenization** | Converts raw strings into token IDs and attention masks. | **Line 653**: `features = self.tokenize(sentences_batch)` |
| **2. Transformer & 3. Pooling** | Runs the neural network forward pass (Self-Attention + Mean Pooling). | **Line 685**: `out_features = self.forward(features, **kwargs)` |
| **Output Extraction** | Extracts the pooled sentence vector tensor. | **Line 713**: `embeddings = out_features[output_value]` |
| **4. L2 Normalization** | Normalizes vector length to 1. | **Line 716**: `embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)` |

---

## 2. The Core Algorithms Explained

### Step 1: Subword Tokenization (WordPiece / Byte-Pair Encoding)
* **Algorithm**: **WordPiece** or **Byte-Pair Encoding (BPE)** (via Hugging Face Tokenizers).
* **Logic**:
  1. Breaks sentences down into subword tokens (e.g. `"embeddings"` $\rightarrow$ `["em", "##bed", "##dings"]`).
  2. Adds special control tokens: `[CLS]` at the beginning and `[SEP]` at the end.
  3. Maps each token to an integer ID according to the model's vocabulary matrix.
  4. Generates an `attention_mask` ($1$ for real text, $0$ for padding tokens).

---

### Step 2: Multi-Head Self-Attention (Transformer Encoder Backbone)
* **Algorithm**: **Scaled Dot-Product Multi-Head Self-Attention** (Vaswani et al., 2017) inside a BERT/RoBERTa architecture (e.g. `thenlper/gte-large`).
* **Mathematical Formula**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
* **Logic**:
  * For every token, the model projects input vectors into Query ($Q$), Key ($K$), and Value ($V$) representations.
  * Tokens cross-attend to every other token in the sequence. This enables each token representation to absorb context from surrounding tokens (e.g., distinguishing "apple" the fruit from "apple" the tech company).
* **Output**: A 3D tensor of shape `(batch_size, sequence_length, hidden_dimension)`. For `gte-large`, each token is mapped to a **1024-dimensional** vector.

---

### Step 3: Pooling (Attention-Masked Mean Pooling)
* **Algorithm**: **Attention-Masked Mean Pooling**.
* **Problem**: A chunk contains multiple tokens (e.g., 100 tokens $\times$ 1024 dimensions), but vector databases (like Pinecone) require **one single fixed-size vector** (1024 dimensions) per chunk.
* **Logic**:
  Instead of only using the first `[CLS]` token, models like `gte-large` average all contextualized token vectors across the sequence while masking out padding:
  $$\vec{v}_{\text{sentence}} = \frac{\sum_{i=1}^{L} \text{token\_emb}_i \times \text{attention\_mask}_i}{\sum_{i=1}^{L} \text{attention\_mask}_i}$$
* **Where it runs**: In `SentenceTransformer.forward()` (Lines 751–759), the model executes a pipeline of sequential modules:
  1. `models.Transformer`: Produces contextual token embeddings.
  2. `models.Pooling`: Aggregates token embeddings into a single sentence embedding using the formula above.

---

### Step 4: L2 Normalization (Euclidean Unit Vector)
* **Algorithm**: **$L_2$ Normalization** (Line 716):
  $$\hat{v} = \frac{\vec{v}}{\|\vec{v}\|_2} = \frac{\vec{v}}{\sqrt{\sum_{j} v_j^2}}$$
* **Logic**:
  Scales the embedding vector to unit Euclidean norm ($\|\hat{v}\| = 1.0$). This ensures that computing Cosine Similarity simplifies to a dot product:
  $$\text{Cosine Similarity}(A, B) = \hat{A} \cdot \hat{B}$$



to see markdown form press Ctrl + Shift + V