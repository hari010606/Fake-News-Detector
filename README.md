# Fake News Detector for Students

An AI-powered platform that helps students verify news authenticity and develop critical media literacy skills using transformer models and similarity search.

![Fake News Detector](https://img.shields.io/badge/Version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)

## Overview

The Fake News Detector for Students provides an AI-driven, easy-to-use platform for verifying news authenticity. By integrating transformer models with vector similarity search, the system efficiently analyzes credibility and helps students distinguish between real and misleading information, promoting digital literacy and responsible media consumption.

## Features

- **AI-Powered Analysis**: Uses DistilBERT model fine-tuned on sentiment analysis
- **Similarity Search**: Compares against database of 180,000+ real/fake news examples
- **Student-Focused**: Designed specifically for educational use cases
- **Confidence Scoring**: Provides transparency with confidence levels
- **Educational Recommendations**: Offers practical media literacy tips
- **Easy-to-Use Interface**: Simple Gradio web interface

## Installation

### Prerequisites
- Python 3.8 or higher
- 2GB+ RAM
- 1GB+ disk space

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
