# 👩🏿 **padie-extended** 👩🏿

[![PyPI version](https://badge.fury.io/py/padie-extended.svg)](https://badge.fury.io/py/padie-extended)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)

**padie-extemded** is an open-source Python package designed to predict Nigerian languages, including **Pidgin**, **Yoruba**, **Hausa**, and **Igbo**. It provides AI-powered tools for **language detection** and fosters community collaboration to enhance its capabilities.

#### **🔧 Note:** 
Padie-extended is a work in progress. It is an extension developed by [Ayooluwaposi Olomo](https://www.linkedin.com/in/ayooluwaposi-olomo-7a322b185/), building upon the original [Padie repository](https://github.com/sir-temi/Padie) by [@sir-temi](https://github.com/sir-temi) and [@pythonisoft](https://github.com/pythonisoft). Their open-source work laid the foundation for this project. Contributions are welcome. Be sure to check out their repository!

---

## Features

- 🚀 **Fast and accurate** language detection for Nigerian languages
- 🤖 **Pre-trained transformer model** for high-quality predictions
- 🌍 **Supports 5 languages**: English, Nigerian Pidgin, Yoruba, Hausa, and Igbo
- 📦 **Simple API** - just a few lines of code
- 🔧 **Easy integration** into existing Python projects
- 💻 **Lightweight and efficient** for production use

---

## 🚫 Dataset Contributions
**Please do NOT submit datasets to this repository.** All dataset contributions 
should be made to the [original Padie repository](https://github.com/sir-temi/Padie). This ensures all 
Padie-based projects benefit from your contributions.

## 🤝 How You Can Contribute:

We welcome contributions from developers, linguists, and data scientists interested in improving Nigerian language technology.

Here are some impactful ways you can help:

- **Expand Language Coverage**:  
  Add support for more Nigerian and African languages beyond those currently included.

- **Improve Short-Form Text Handling**:  
  The model performs better on long-form text. Training and fine-tuning it on short-form (social media, chat, etc.) data can boost performance.

- **Optimize Inference Efficiency**:  
  Reduce model size or latency for deployment on resource-limited environments (mobile, low-bandwidth servers).

- **Enhance Evaluation Metrics**:  
  Add multilingual or domain-specific benchmarks (e.g., dialectal variations, code-switching).

- **Augment the Dataset**:  
  Contribute curated, diverse, and balanced text data to the **main Padie repository**, not this one.

- **Improve Documentation & Examples**:  
  Add usage examples, Jupyter notebooks, or tutorials showing real-world use cases.

---

### 🧠 Quick Contribution Steps

1. **Fork the Repository**:  
   Click the "Fork" button at the top of the repository page to create your copy.

2. **Clone Your Fork**:

    ```bash
    git clone https://github.com/sir-temi/Padie.git
    ```

3. **Create a Branch**:

    ```bash
    git checkout -b feature-name
    ```

4. **Make Your Changes**:

   - Model improvements and training techniques
   - Bug fixes and code optimizations
   - Documentation and examples
   - Evaluation tools and metrics

5. **Commit and Push**:

    ```bash
    git commit -m "Describe your changes"
    git push origin feature-name
    ```

6. **Submit a Pull Request**:  
   Open a pull request against the `dev` branch with a clear description of your changes.

---

## 📦 Installation

```bash
pip install padie-extended
```

## ⌛ Quick Start

```python
from padie_extended import LanguageDetector

# Initialize the detector
detector = LanguageDetector()

# Detect language from text
text = "Bawo ni, se daadaa ni?"
result = detector.predict(text)

print(f"Language: {result['language']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Output:
```
Language: Yoruba
Confidence: 98.50%
```

## 🌍 Supported Languages

| Language | Code | Example |
|----------|------|---------|
| English | `en` | "Hello, how are you?" |
| Nigerian Pidgin | `pidgin` | "How you dey?" |
| Yoruba | `yo` | "Bawo ni?" |
| Hausa | `ha` | "Sannu" |
| Igbo | `ig` | "Kedu?" |

## 💡Usage Examples

### Basic Detection

```python
from padie_extended import LanguageDetector

detector = LanguageDetector()

# Single text
text = "I dey kampe, na God"
result = detector.predict(text)
print(result)
# {'language': 'pidgin', 'all_scores': {...}, 'confidence': 0.96}
```

### Batch Processing

```python
texts = [
    "Good morning everyone",
    "Ẹ káàárọ̀",
    "Sannu da safe",
    "Wetin dey happen?"
]

results = detector.predict_batch(texts)
for text, result in zip(texts, results):
    print(f"{text} -> {result['language']}")
```

### Get All Confidence Scores

```python
result = detector.predict("This is a mixed text")
print(result['all_scores'])
# {
#     'english': 0.85,
#     'pidgin': 0.10,
#     'yoruba': 0.03,
#     'hausa': 0.01,
#     'igbo': 0.01
# }
```

## 🧠Advanced Usage

### Custom Model Path

```python
detector = LanguageDetector(model_path="path/to/your/model")
```

### Custom Confidence Threshold

```python
# Set threshold at initialization (default is 0.5)
detector = LanguageDetector(confidence_threshold=0.7)

# Or override for a specific prediction
result = detector.predict("Maybe pidgin", threshold=0.8)

# Change threshold after initialization
detector.set_threshold(0.6)
```

## Model Information

- **Base Model**: [afro-xlmr-base](https://huggingface.co/Davlan/afro-xlmr-base) Transformer-based model
- **Training Data**: Diverse corpus of Nigerian language texts
- **Model Size**: 1GB

## Performance

Tested on a diverse dataset of Nigerian texts:

| Metric | Score |
|--------|-------|
| Overall Accuracy | 95.3% |
| F1 Score (weighted) | 95.3% |
| Inference Speed | ~4.5 ms per text (measured on GPU) |


## Requirements [End User]
If you’re using this package to detect languages in your own projects (not for model training or development), you only need the following dependencies:
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- SentencePiece 0.1.99+

## Use Cases

- 🌐 **Content moderation** - Detect language in user-generated content
- 📱 **Social media analysis** - Analyze multilingual Nigerian social media posts
- 🤖 **Chatbots** - Route conversations based on detected language
- 📊 **Research** - Analyze language distribution in datasets
- 🎯 **Language-specific processing** - Trigger different pipelines per language

---

## Citation

If you use this package in your research, please cite:

```bibtex
@software{padie_extended,
  author = {Olomo, Ayooluwaposi},
  title = {padie-extended: AI-powered Nigerian Language Detection},
  year = {2025},
  url = {https://github.com/posi-olomo/padie-extended}
}
```

## Acknowledgments

- Built upon the [Padie](https://github.com/sir-temi/Padie) project
- Built with AWS cloud credits generously provided by Dr. Wálé Akínfadérìn
- Built with [Hugging Face Transformers](https://huggingface.co/transformers/)
- Inspired by the need for better Nigerian language NLP tools
- Thanks to all future contributors and the Nigerian NLP community

## Links

- **GitHub**: [posi-olomo/padie-extended](https://github.com/posi-olomo/padie-extended)
- **PyPI**: [padie-extended](https://pypi.org/project/padie-extended/)
- **Issues**: [Report a bug](https://github.com/posi-olomo/padie-extended/issues)
- **Documentation**: [Full Documentation](https://github.com/posi-olomo/padie-extended/wiki)

## Support

If you encounter any issues or have questions:

1. Check the [documentation](https://github.com/posi-olomo/padie-extended/wiki)
2. Search [existing issues](https://github.com/posi-olomo/padie-extended/issues)
3. Create a [new issue](https://github.com/posi-olomo/padie-extended/issues/new)

---

## 🌍 **Open Source Contribution**

**padie-extended** is licensed under the [MIT License](https://opensource.org/licenses/MIT), ensuring it remains free and open for everyone to use, contribute to, and enhance.



**Made with ❤️ for the Nigerian tech community**