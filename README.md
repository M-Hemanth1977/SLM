# CGT-Bot: GNN-Enhanced Encoder-Decoder for RAG on Fed-Meta-Align

## Overview
This project implements **CGT-Bot**, a custom Graph Neural Network (GNN)-augmented Transformer Encoder-Decoder model designed for Retrieval-Augmented Generation (RAG) tasks. The model is fine-tuned on a PDF document about **Fed-Meta-Align** (a federated TinyML framework for IoT devices) and enriched with Q&A pairs. It leverages **GPT-2 embeddings** for initialization and incorporates a **GNN layer** for capturing local token dependencies in sequences.

**Key features:**
- Pretraining on Wikipedia text for general language understanding.
- Domain adaptation on the Fed-Meta-Align document.
- Instruction fine-tuning on synthetic Q&A pairs.
- RAG Integration using SentenceTransformer embeddings for efficient retrieval.
- Deployment-ready: Independent query processing (no conversation history) with intelligent answer extraction.
- Achieves low perplexity through multi-stage training and supports edge-device inference (e.g., IoT fault classification).

---

## Architecture Details
**GNNEncoderDecoder hybrid architecture** combining:
- **Token Embeddings:** GPT-2 initialized (50257 vocab, 768-dim embeddings)
- **Positional Embeddings:** Learned embeddings for sequence positions (up to 512 tokens)
- **GNN Module:** Captures local syntactic dependencies
- **Transformer Encoder:** Global context modeling
- **Transformer Decoder:** Autoregressive generation with causal masking
- **Output Projection:** Tied to input embeddings for efficiency

**Configuration (CGTConfig)**

| Parameter | Value | Description |
|-----------|-------|-------------|
| vocab_size | 50257 | GPT-2 vocabulary size |
| hidden_dim | 768 | Embedding and hidden state dimension |
| gnn_layers | 2 | Number of GNN layers (GATv2Conv) |
| transformer_encoder_layers | 4 | Encoder Transformer layers |
| transformer_decoder_layers | 2 | Decoder Transformer layers |
| num_heads | 8 | Multi-head attention heads |
| gnn_type | 'gat' | Graph Attention Network (GATv2) |
| dropout | 0.1 | Dropout rate across layers |
| max_seq_len | 512 | Maximum input sequence length |

**Model Layers Breakdown:**
- **Input Embeddings:** Size `(50257, 768)`, pre-loaded from GPT-2. Adds positional embeddings `(512, 768)`.
- **GNN Layers:** 2 GATv2 layers with LayerNorm, ReLU, and Dropout. Tokens as nodes; edges include bidirectional chains + skip connections.
- **Encoder:** 4 layers TransformerEncoderLayer, `d_model=768, nhead=8, dim_feedforward=3072`.
- **Decoder:** 2 layers TransformerDecoderLayer, causal masking + cross-attention to encoder memory.
- **Output Layer:** Projects decoder outputs to vocab (50257-dim logits), weight-tying with input embeddings.

**Parameters Summary:**
- Total ~88.7M
  - Embeddings: ~38.5M  
  - GNN: ~1.2M  
  - Encoder: ~33.2M  
  - Decoder: ~16.6M  
  - Output: tied  

---

## Training Pipeline
**Stages:**
1. **Pretraining (Language Modeling)**
   - Dataset: 10,000 Wikipedia samples
   - Epochs=3, LR=1e-4, Batch=16
   - Loss: 3.45 → 2.52
   - Purpose: Learn English patterns

2. **Domain Fine-Tuning**
   - Dataset: 276 sentences from Fed-Meta-Align PDF
   - Epochs=10, LR=5e-5, Batch=8
   - Loss: 2.89 → 2.12
   - Purpose: Adapt to technical domain

3. **Instruction Fine-Tuning (Q&A)**
   - Dataset: 548 samples (276 domain + 272 Q&A)
   - Epochs=50, LR=3e-5, Batch=8
   - Loss: 2.34 → 0.64
   - Purpose: Enable Q&A and instruction-following

---

## RAG System Integration
- **Retriever:** SentenceTransformer (all-MiniLM-L6-v2) on 142 document chunks
- **Generation:** Beamless sampling (`temp=0.7, top-k=50, repetition_penalty=1.3`)
- **Post-Processing:** Intelligent extraction (gibberish detection, sentence filtering)
- **Fallback:** Extractive QA from top-2 chunks if generative output fails

---

