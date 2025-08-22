# SLM-RAG Pipeline

This repository contains a Google Colab notebook **`slm_rag.ipynb`** which demonstrates how to build, run, and experiment with a **Small Language Model (SLM)** integrated with a **Retrieval-Augmented Generation (RAG)** pipeline. 

---

## Project Overview

The **SLM-RAG Complex Pipeline** demonstrates how to integrate a **Small Language Model (SLM)** with a **Retrieval-Augmented Generation (RAG)** framework for efficient information retrieval and generation in a resource-constrained environment like Google Colab.  

The pipeline focuses on reading, preprocessing, embedding, retrieving, and generating text responses from large-scale custom datasets such as technical guides or Wikipedia text.

---

### Key Components

1. **Data Access and Integration**
   - Links Colab to Google Drive for seamless access to large text files (`wiki.train.tokens`, `extracted_text.txt`).
   - Supports flexible reading strategies (by characters, lines, or paragraphs).

2. **Text Preprocessing**
   - Uses **NLTK** for:
     - Sentence tokenization (`sent_tokenize`)
     - Word tokenization (`word_tokenize`)
     - Stopword handling
   - Extracts meaningful text samples from raw files for fine-tuning and retrieval.

3. **Embedding Generation**
   - Employs **SentenceTransformers (all-MiniLM-L6-v2)** to encode text into dense vector embeddings.
   - Provides semantic representation of text chunks to enable similarity-based retrieval.

4. **Tokenizer and Model Setup**
   - Uses **GPT-2 tokenizer** (`GPT2Tokenizer`) for preparing input text.
   - Sets tokenizer padding token as EOS to handle varying sequence lengths.
   - Prepares tokenized data for fine-tuning and inference.

5. **Document Chunking**
   - Splits input text into structured **paragraphs** and **chunks**.
   - Filters out short or irrelevant segments to keep only meaningful content.
   - These chunks are later indexed for retrieval in the RAG pipeline.

6. **Fine-Tuning Preparation**
   - Processes sentences into structured fine-tuning samples.
   - Ensures that only sufficiently long and clean sentences are included.
   - Generates training-ready datasets for downstream tasks.

7. **RAG Workflow**
   - **Retrieval**: Uses vector embeddings (via SentenceTransformers) for similarity search.
   - **Augmentation**: Retrieves top-k relevant passages from the indexed dataset.
   - **Generation**: Passes retrieved passages to a Small Language Model for context-grounded text generation. 

---

## Detailed Theory of the Architecture

This architecture is designed to combine the strengths of **Graph Neural Networks (GNNs)** and **Transformers**. The idea is to capture both **structural relationships** (from the graph perspective) and **contextual dependencies** (from sequence modeling). The flow of the architecture is as follows:

---

### 1. Input & Initial Embedding Layer
The raw input features are first converted into dense vector representations. This step ensures that the data is represented in a form suitable for both graph-based and transformer-based processing.

---

### 2. Graph Neural Network (GNN) Layer
The embeddings are passed into a GNN, which captures the **local structural information** by allowing each node to learn from its neighbors. This is especially useful when the data has underlying graph relationships, such as dependencies, connections, or interactions between entities.

---

### 3. Normalization Layer
Before moving to the Transformer layers, normalization is applied to stabilize training. It ensures that feature distributions are consistent and prevents issues like exploding or vanishing gradients.

---

### 4. Transformer Encoder Layers
Next, the normalized features are processed through Transformer encoder layers. This stage captures **long-range dependencies and contextual information** across the entire input sequence. Unlike GNNs that focus on local neighborhoods, Transformers are powerful at modeling global relationships.

---

### 5. Fusion Layer
The outputs from the **GNN** and the **Transformer** are then fused together. This step combines the advantages of both:
- The **GNN** contributes structural, neighborhood-based information.  
- The **Transformer** contributes contextual and sequential information.  

By fusing them, the model creates a richer representation that balances both perspectives.

---

###  6. Re-injection into Transformer
The fused representation is fed back into an additional Transformer layer. This refinement step allows the model to **re-attend to the combined features** and enhance its understanding by redistributing attention to the most relevant parts of the fused data.

---

###  7. Linear Output Layer
Finally, the refined features are passed through a linear output layer (with softmax for classification or a regression head for numeric predictions). This produces the final model outputs.

---

###  Why This Design?
1. **GNN first** – captures local and relational structure early on.  
2. **Transformer next** – captures long-range dependencies across the sequence or graph.  
3. **Fusion step** – integrates graph-based and sequence-based features.  
4. **Reinjection into Transformer** – refines the fused features by redistributing attention.  
5. **Linear head** – produces the final predictions in a task-specific manner.  

---

### Benefits of the Architecture
- **Rich feature representations** by combining structural and sequential knowledge.  
- **Multi-level understanding** with both local and global perspectives.  
- **Preserves GNN knowledge** by fusing and reinjecting into the Transformer rather than letting it get overshadowed.  
- **Flexibility** to handle different kinds of inputs (graph-structured, sequential, or hybrid).  

