# 🤖 GenAI Learning Repository

A comprehensive learning journey through Generative AI, covering Python fundamentals, Natural Language Processing, Deep Learning, and hands-on mini projects using LangChain and HuggingFace.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-red.svg)](https://streamlit.io/)

---

## 📚 Table of Contents

- [Learning Path](#-learning-path)
  - [Python & OOP](#-python--oop)
  - [NLP Fundamentals](#-nlp-fundamentals)
  - [Deep Learning](#-deep-learning)
- [Mini Projects](#-mini-projects)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)

---

## 🎯 Learning Path

### 🐍 Python & OOP

📂 [View Notebooks](./Hands-On-Notebooks/1.OOPS) | 📂 [Streamlit Projects](./Hands-On-Notebooks/2.STREAMLIT)

#### **Object-Oriented Programming**

<details>
<summary><b>Core OOP Concepts</b></summary>

- **OOP Basics**: Classes, objects, attributes, methods, constructors
- **Inheritance**: Parent/child classes, method overriding, super keyword
- **Polymorphism**: Method overloading, method overriding, Abstract Base Class
- **Encapsulation**: Private/protected variables, getters & setters
- **Abstraction**: Abstract classes, abstract methods, ABC module
- **Magic Methods**: `__init__`, `__str__`, `__repr__`, `__len__`, `__getitem__`
- **Operator Overloading**: Arithmetic & comparison operators

</details>

#### **Streamlit Framework**

- Building interactive web applications
- Widgets and user inputs
- Data visualization
- ML/AI app interfaces

---

### 📝 NLP Fundamentals

📂 [View Notebooks](./Hands-On-Notebooks/3.NLP(Word%20Embedding))

#### **Text Preprocessing**

| Topic | Concepts Covered |
|-------|-----------------|
| **Tokenization** | Word, sentence, and subword tokenization |
| **Stemming** | Porter, Regexp, Snowball stemmers |
| **Lemmatization** | WordNet Lemmatizer, comparison with stemming |
| **Stopwords** | Removal techniques and applications |
| **POS Tagging** | Part-of-speech tagging with NLTK |
| **NER** | Named Entity Recognition, chunking, tree structures |

#### **Word Embedding Techniques**

| Technique | Description |
|-----------|-------------|
| **One Hot Encoding** | Binary vector representation |
| **Bag of Words** | Document-term frequency matrix |
| **N-Grams** | Bigrams, trigrams for context |
| **TF-IDF** | Term frequency-inverse document frequency |
| **Word2Vec** | Dense word embeddings, CBOW, Skip-gram |

---

### 🧠 Deep Learning

📂 [View Notebooks](./Hands-On-Notebooks/4.DEEP-LEARNING)

#### **Recurrent Neural Networks (RNN)**

<details>
<summary><b>Simple RNN</b></summary>

- **ANN vs RNN**: Understanding sequential data challenges
- **Simple RNN**: Architecture, forward/backward propagation
- **Vanishing Gradient Problem**: Challenges with long sequences and why it matters

</details>

#### **Long Short-Term Memory (LSTM)**

<details>
<summary><b>LSTM Architecture & Components</b></summary>

📂 [LSTM Learning Materials](./Hands-On-Notebooks/4.DEEP-LEARNING/LSTM-RNN) | 🖼️ [Reference Images](./Hands-On-Notebooks/4.DEEP-LEARNING/LSTM-RNN/images)

**Why LSTM?**
- Solves the vanishing gradient problem of simple RNNs
- Enables learning from long-term dependencies
- Maintains information across long sequences

**Core Components:**

| Component | Description |
|-----------|-------------|
| **Cell State (Ct)** | The "memory" of the network, carries information across time steps |
| **Hidden State (ht)** | The output of the LSTM cell at each time step |
| **Forget Gate** | Decides what information to discard from the cell state |
| **Input Gate** | Decides what new information to store in the cell state |
| **Output Gate** | Decides what information from the cell state to output |

**Gate Operations:**

1. **Forget Gate (ft)**: `ft = σ(Wf · [ht-1, xt] + bf)`
   - Uses sigmoid activation to output values between 0 and 1
   - 0 = completely forget, 1 = completely keep

2. **Input Gate (it)**: `it = σ(Wi · [ht-1, xt] + bi)`
   - Decides which values to update in the cell state
   - Works with candidate values: `C̃t = tanh(Wc · [ht-1, xt] + bc)`

3. **Cell State Update**: `Ct = ft * Ct-1 + it * C̃t`
   - Combines forget and input operations
   - Maintains long-term memory

4. **Output Gate (ot)**: `ot = σ(Wo · [ht-1, xt] + bo)`
   - Decides what to output based on cell state
   - Final output: `ht = ot * tanh(Ct)`

**Key Advantages:**
- Better gradient flow during backpropagation
- Selective memory (can choose what to remember/forget)
- Handles long-term dependencies effectively
- Widely used in NLP, time series, and sequence modeling

</details>

#### **LSTM Variants**

<details>
<summary><b>Different Flavours of LSTM</b></summary>

📂 [LSTM Variants Notebook](./Hands-On-Notebooks/4.DEEP-LEARNING/LSTM-RNN/4.14-Variants-LSTM.ipynb)

| Variant | Key Difference |
|---------|----------------|
| **Vanilla LSTM** | Standard LSTM with all three gates |
| **Peephole LSTM** | Gates can peek at cell state `Ct-1` directly |
| **Coupled Gates** | Forget and input gates are linked — what is forgotten is replaced by new info |
| **GRU (Gated Recurrent Unit)** | Merges cell state & hidden state; only 2 gates (Update + Reset) |

</details>

#### **Gated Recurrent Unit (GRU)**

<details>
<summary><b>GRU Architecture & Comparison with LSTM</b></summary>

📂 [GRU Notebook](./Hands-On-Notebooks/4.DEEP-LEARNING/LSTM-RNN/4.15-GRU-RNN.ipynb)

**Why GRU?**
- Simpler than LSTM — fewer parameters, faster to train
- Combines forget & input gates into a single **Update Gate**
- Merges cell state and hidden state into one

**Gate Mechanisms:**

| Gate | Formula | Role |
|------|---------|------|
| **Update Gate (zt)** | `zt = σ(Wz · [ht-1, xt])` | Controls how much past info to keep |
| **Reset Gate (rt)** | `rt = σ(Wr · [ht-1, xt])` | Controls how much past info to forget |
| **Candidate State** | `h̃t = tanh(W · [rt * ht-1, xt])` | New candidate hidden state |
| **Output** | `ht = (1-zt) * ht-1 + zt * h̃t` | Final hidden state |

**GRU vs LSTM:**

| | GRU | LSTM |
|--|-----|------|
| **Gates** | 2 (Update, Reset) | 3 (Forget, Input, Output) |
| **States** | 1 (Hidden state) | 2 (Cell + Hidden state) |
| **Parameters** | Fewer | More |
| **Speed** | Faster | Slower |
| **Best for** | Smaller datasets, faster training | Long sequences, complex tasks |

</details>

#### **Bidirectional RNN**

<details>
<summary><b>Bidirectional Architecture & Use Cases</b></summary>

📂 [Bidirectional RNN Notebook](./Hands-On-Notebooks/4.DEEP-LEARNING/Bidirectional-RNN/Bidir.ipynb)

**What is Bidirectional RNN?**
- Runs two RNNs simultaneously — one forward (past → future) and one backward (future → past)
- Final output is a combination of both directions
- Can be applied to any RNN variant: Simple RNN, LSTM, GRU

**How it works:**
```
Forward RNN:   x1 → x2 → x3 → x4  (left to right)
Backward RNN:  x4 → x3 → x2 → x1  (right to left)
Output:        concat(forward_ht, backward_ht)
```

**Key Advantages:**
- Captures context from **both past and future** tokens
- Significant improvement for NLP tasks (translation, NER, sentiment)
- Foundation for modern architectures like BERT

**When to use:**
- ✅ NLP tasks where full-sentence context matters
- ✅ Named Entity Recognition (NER)
- ✅ Machine Translation
- ❌ Time series / real-time prediction (future data unavailable)

</details>

---

## 🚀 Mini Projects

📂 [View All Projects](./Mini-projects)

### 1️⃣ Customer Churn Prediction 📊

> **Binary Classification** using Artificial Neural Networks

**🎯 Objective:** Predict whether a customer will leave the bank based on their profile and behavior.

**✨ Features:**
- 🖥️ Interactive Streamlit web interface
- 🧠 Deep Learning (ANN) for binary classification
- 📈 Real-time prediction with probability scores
- 📊 Customer demographics, credit score, balance, products

**🔧 Tech Stack:**
- TensorFlow/Keras
- Streamlit
- scikit-learn (StandardScaler, LabelEncoder, OneHotEncoder)

**📁 Project Structure:**
```
CHURN-MODELLING/
├── app.py                    # Streamlit application
├── experiments.ipynb         # Model training & experimentation
├── prediction.ipynb          # Model evaluation
├── model.h5                  # Trained model
└── *.pkl                     # Preprocessing artifacts
```

---

### 2️⃣ Salary Regression Predictor 💰

> **Regression Model** to estimate customer salary

**🎯 Objective:** Predict estimated annual salary based on customer banking profile.

**✨ Features:**
- 🖥️ Beautiful Streamlit UI with metric displays
- 🧠 ANN Regression model
- 📊 Customer profile summary visualization
- 📈 TensorBoard integration for training monitoring

**🔧 Tech Stack:**
- TensorFlow/Keras
- Streamlit
- TensorBoard
- scikit-learn (StandardScaler, LabelEncoder, OneHotEncoder)

**📁 Project Structure:**
```
Regression/
├── streamlit_reg.py          # Streamlit application
├── Salaryregression.ipynb    # Model training notebook
├── regression_model.h5       # Trained regression model
├── logs/                     # TensorBoard logs
└── *.pkl                     # Preprocessing artifacts
```

---

### 3️⃣ Movie Review Sentiment Analysis 🎬

> **RNN-based Sentiment Classification** for IMDB movie reviews

**🎯 Objective:** Classify movie reviews as positive or negative using Recurrent Neural Networks.

**✨ Features:**
- 🖥️ Clean and intuitive Streamlit interface
- 🧠 Simple RNN model trained on IMDB dataset
- 📊 Real-time sentiment prediction with confidence scores
- 🎭 Handles user-provided movie reviews of any length
- 📈 Preprocessing pipeline with word embedding

**🔧 Tech Stack:**
- TensorFlow/Keras (SimpleRNN)
- Streamlit
- IMDB Dataset (10,000 vocabulary size)
- Sequence padding (max length: 500)

**📁 Project Structure:**
```
Movie-Review-RNN/
├── main.py                   # Streamlit application
├── RnnProject.ipynb          # Model training notebook
├── prediction.ipynb          # Model evaluation & testing
└── simple_rnn_imdb.keras     # Trained RNN model
```

---

### 4️⃣ Next Word Prediction with LSTM 📝

> **LSTM-based Text Generation and Word Prediction** | [🚀 Live Demo](https://wordpred.streamlit.app/)

**🎯 Objective:** Predict the next word in a sequence using LSTM trained on Shakespeare's Hamlet.

**✨ Features:**
- 🌐 Deployed live on Streamlit Cloud
- 🧠 LSTM model for sequence prediction
- 📚 Trained on classic literature (Shakespeare's Hamlet)
- ⚡ Real-time word prediction
- 🎯 Early stopping for optimal training
- 🔮 Handles variable-length input sequences

**🔧 Tech Stack:**
- TensorFlow/Keras (LSTM)
- Streamlit
- Text Tokenization & Preprocessing
- Sequence Padding

**📁 Project Structure:**
```
LSTM-RNN/
├── app.py                    # Streamlit application
├── experiments.ipynb         # Model training & experimentation
├── hamlet.txt                # Training dataset (Shakespeare)
├── word_prediction_lstm.h5   # Trained LSTM model
└── tokenizer.pickle          # Text tokenizer
```

**🌐 Live Demo:** [https://wordpred.streamlit.app/](https://wordpred.streamlit.app/)

---

## 🛠️ Tech Stack

### **Core Technologies**

| Category | Tools & Libraries |
|----------|------------------|
| **Languages** | Python 3.10+ |
| **Deep Learning** | TensorFlow 2.15, Keras |
| **ML Libraries** | scikit-learn, NumPy, Pandas |
| **NLP** | NLTK, Gensim |
| **Web Framework** | Streamlit 1.52 |
| **Visualization** | Matplotlib, Seaborn, TensorBoard |
| **Version Control** | Git, GitHub |

---

## 🚦 Getting Started

### Prerequisites

```bash
Python 3.10 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sukrit-89/genai-playground-langchain-hf.git
   cd genai-playground-langchain-hf
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running Mini Projects

#### Churn Prediction App
```bash
cd Mini-projects/CHURN-MODELLING
streamlit run app.py
```

#### Salary Regression App
```bash
cd Mini-projects/Regression
streamlit run streamlit_reg.py
```

#### Movie Review Sentiment Analysis
```bash
cd Mini-projects/RNN/Movie-Review-RNN
streamlit run main.py
```

#### Next Word Prediction (LSTM)
```bash
cd Mini-projects/LSTM-RNN
streamlit run app.py
```

**🌐 Or visit the live deployment:** [https://wordpred.streamlit.app/](https://wordpred.streamlit.app/)

---

## 📈 Learning Progress

- ✅ Python OOP Fundamentals
- ✅ Streamlit Framework
- ✅ NLP Text Preprocessing
- ✅ Word Embedding Techniques
- ✅ Deep Learning Basics (ANN, RNN)
- ✅ LSTM-RNN Architecture & Training Process
- ✅ LSTM Variants (Peephole, Coupled Gates)
- ✅ GRU (Gated Recurrent Unit)
- ✅ Bidirectional RNN
- ✅ Binary Classification Project (Churn Prediction)
- ✅ Regression Project (Salary Estimation)
- ✅ RNN Sentiment Analysis Project (Movie Reviews)
- ✅ LSTM Word Prediction Project (Next Word Generation)
- 🔄 Transformers & Attention Mechanism - In Progress

---

## 📝 Notes

- All notebooks are organized by topic and concept
- Model files (`.h5`, `.pkl`) are excluded from version control
- TensorBoard logs available for training visualization
- Datasets are not tracked in Git (see `.gitignore`)

---

## 🤝 Contributing

This is a personal learning repository. However, suggestions and feedback are always welcome!

---

## 📄 License

This project is for educational purposes.

---

<div align="center">

**Happy Learning! 🚀**

Made with ❤️ by Sukrit

</div>
