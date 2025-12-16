# 🛰️ Space Traffic Management: LEO Orbit Prediction with XGBoost, LightGBM, and FT-Transformer

<div align="center">

![Project Banner](https://img.shields.io/badge/Project%20Type-Data%20Science%2FML-blue.svg?style=for-the-badge)
[![GitHub stars](https://img.shields.io/github/stars/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge&logo=github)](https://github.com/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge&logo=git)](https://github.com/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer/network)
[![GitHub issues](https://img.shields.io/github/issues/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge&logo=github)](https://github.com/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer/issues)
[![GitHub license](https://img.shields.io/github/license/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer?style=for-the-badge)](LICENSE)

**Leveraging advanced machine learning for robust prediction and management of Low Earth Orbit (LEO) space traffic.**

</div>

## 📖 Overview

Abstract
Given the increasing presence of satellites and debris within Low Earth Orbit, there clearly exists a need for early indications on high-risk conjunctions, and we undertook this research endeavor as an exploration into utilizing models based on machine learning. We processed a set combining CDM files for 574,289 data instances with structured data cleansing, removal of leakage variables, and added engineered variables. We then developed models based on three criteria: an FT-Transformer model based on original clean variables and boolean flags, LightGBM and XGBoost based on datasets with variables, and additional models without leakage for determining which extent models rely on variables ‘cdmPc’ and ‘miss distance’. We normalized hyperparameters and weights for ensembles via Optuna with thresholds determined via scanning probabilities for maximum recall while remaining at precision values exceeding 0.50. The FT-Transformer model demonstrated superior solo performance with precision and recall equaling 1.00 on multiple instances on the testing set, indicating successful learning on CDM data via an attention-based model. The ensemble model combining predictions for all XGBoost and LightGBM models offered consistent predictions for various attribute forms and satisfied the precision constraint at high recall rates. Outcomes indicate that respective transformer models and weighted ensembles based on tree models could be implemented effectively for high-risk conjunction detection with an optimal precision-recall band via thresholding.

**Keywords:** Space Traffic Management; Conjunction Assessment; Machine Learning; FT‑Transformer; XGBoost; LightGBM; Collision Risk Prediction

## ✨ Features

-   **Multi-Model Comparative Analysis:** Implements and compares the predictive performance of XGBoost, LightGBM, and FT-Transformer models for LEO STM.
-   **Targeted LEO Focus:** Addresses the unique challenges and high-density environment characteristic of Low Earth Orbit.
-   **Data Processing Pipeline:** Includes stages for loading raw competition data, essential preprocessing, and feature engineering to prepare data for diverse ML architectures.
-   **Robust Model Training & Evaluation:** Demonstrates the full lifecycle of model development, including training, hyperparameter tuning insights, and rigorous evaluation using appropriate metrics within a reproducible notebook.
-   **Deep Learning for Tabular Data:** Explores the application of the FT-Transformer, a state-of-the-art neural network, for superior feature learning from structured data.
-   **Pre-trained Model Integration:** Includes pre-trained weights (`ft_transformer.pth`) for the FT-Transformer, allowing for direct evaluation or fine-tuning.

## 🛠️ Tech Stack

**Primary Language:**
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

**Machine Learning & Data Science Libraries:**
[![XGBoost](https://img.shields.io/badge/XGBoost-005101?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4169E1?style=for-the-badge&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-BA3B0C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-328699?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)

**Development Tools:**
[![VS Code](https://img.shields.io/badge/VS%20Code-0078D4?style=for-the-badge&logo=visualstudiocode&logoColor=white)](https://code.visualstudio.com/)
[![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

## 🚀 Quick Start

To set up this project and run the orbital traffic management experiments, follow these instructions.

### Prerequisites

-   **Python 3.8+** (or newer, as recommended by the ML libraries)
-   **pip** (Python package installer, usually bundled with Python)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/ShivJee-Yadav/Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer.git
    cd Space_Traffic_Management_LEO_Orbit_XGBoost_LightGBM_FT-Transformer
    ```

2.  **Create and activate a virtual environment (recommended)**
    Using a virtual environment helps manage dependencies for different projects without conflicts.
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**
    please install the necessary libraries manually. It is highly recommended to install all dependencies after setting up your environment for future reproducibility.
    ```bash
    pip install pandas numpy scikit-learn xgboost lightgbm torch matplotlib seaborn jupyterlab
    ```
    *Note: If you are using Google Colab, many of these libraries (e.g., NumPy, Pandas, Scikit-learn, PyTorch) are often pre-installed. You might only need to install `xgboost` and `lightgbm` if they are not already available.*

4.  **Prepare the dataset**
    The raw dataset for the competition is provided as `sa-competition-files.zip`. You need to unzip it to access the data. A common practice is to extract it into a dedicated `data/` directory.
    ```bash
    mkdir -p data # Create a 'data' directory if it doesn't exist
    unzip sa-competition-files.zip -d data/
    ```

## 📁 Project Structure

```
.
├── .gitignore                          # Standard Git ignore file
├── IGOM_ML_LightGBM_Colab.ipynb      # Main Jupyter notebook containing ML experiments and analysis
├── README.md                           # The project's README file
├── ft_transformer.pth                  # Pre-trained weights for the FT-Transformer model
├── models/                             #  Directory intended for saving trained models (e.g., .pkl, .pt)
├── outputs/                            #  Directory intended for generated outputs like predictions or intermediate data
├── results/                            #  Directory intended for experimental results, metrics, and plots
├── sa-competition-files.zip            #  Zipped raw dataset for the space traffic management competition
├── src/                                #  all Models source code, utility scripts, or custom modules
└── predict_XGB.py                      #  Predict Files for XG_Boost model
└── predict_LGBM.py                     #  Predict Files for LightGBM model
└── predict_FTTransformer.py            #  Predict Files for FTTTransformer model
```

## ⚙️ Configuration

All model configurations, hyperparameter settings, data preprocessing steps, and experimental parameters are defined directly within the `IGOM_ML_LightGBM_Colab.ipynb` notebook. There are no external configuration files (e.g., `.env`, `config.yaml`) or specific environment variables explicitly detected and used by the project.

## 📊 Results & Models

-   **Pre-trained FT-Transformer:** The `ft_transformer.pth` file contains pre-trained weights for the FT-Transformer model. This allows for immediate loading and use for inference, or as a starting point for further fine-tuning.
-   **Model Storage (`models/`):** This directory is designated for saving trained instances of XGBoost, LightGBM, and FT-Transformer after experimentation.
-   **Output Data (`outputs/`):** Expected to contain generated predictions, processed intermediate datasets, or other files produced during the notebook execution.
-   **Experiment Results (`results/`):** This directory is where evaluation metrics, comparative plots, and other quantitative outcomes of the experiments should be stored.


## 🤝 Contributing

We welcome contributions to further enhance this Space Traffic Management project! Whether you aim to improve model performance, integrate new algorithms, refine data processing, or enhance documentation, your efforts are appreciated. Please refer to these general guidelines:

1.  **Fork** this repository.
2.  **Clone** your forked repository to your local machine.
3.  Create a new **branch** (`git checkout -b feature/your-feature-name`).
4.  Make your changes, ensuring code is well-commented and clear.
5.  **Commit** your changes (`git commit -m 'feat: Add new feature X'`).
6.  **Push** to your branch (`git push origin feature/your-feature-name`).
7.  Open a **Pull Request** to the `main` branch of this repository, describing your contributions.

## 📄 License

This project is licensed under the [LICENSE_NAME](LICENSE). Please see the `LICENSE` file for full details.
*(Note: A `LICENSE` file was not found in the repository. Please add a `LICENSE` file to specify the terms under which your project can be used, distributed, and modified.)*

## 🙏 Acknowledgments

-   To the creators and maintainers of **XGBoost**, **LightGBM**, **PyTorch**, **scikit-learn**, **Pandas**, and **NumPy** for providing powerful open-source tools that are foundational to this project.
-   The broader scientific community and space agencies for their ongoing work in advancing Space Traffic Management.


<div align="center">

**⭐ If this project aids your understanding or work in Space Traffic Management, please consider starring the repository!**

Made with NexusIITJ

</div>