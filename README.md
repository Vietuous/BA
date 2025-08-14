# Comprehensive Project Reproduction Guide in VS Code

---

## Full Project Reproduction Guide (VS Code)

This README serves as a **comprehensive, step-by-step reproduction manual** for setting up and running the project from scratch.  
It provides:
- Context for **why** each step is necessary  
- OS-specific instructions  
- Troubleshooting tips  
- Final verification steps to ensure reproducibility  

> A shortened version can be found directly in a markdown cell inside `reddit_scraper.ipynb` between the documentation (Updates) and the actual code.

---

Before starting, ensure the following are installed:

- **Visual Studio Code** – [Download](https://code.visualstudio.com/)
- **Python 3.11.9** – [Download](https://www.python.org/downloads/release/python-3119/)  
  *(Newer versions may work but were not tested for this project.)*
- **Git** *(optional, for cloning the repository)* – [Download](https://git-scm.com/downloads)

> **Why Python 3.11.9?**
> Some ML dependencies (e.g., XGBoost, SHAP) are more stable on this version. The fixes made here *may* allow newer versions, but they are not guaranteed to work.

---
 
## 2. Prerequisites

Before starting, ensure the following are installed:

- **Visual Studio Code** – [Download](https://code.visualstudio.com/)
- **Python 3.11.9** – [Download](https://www.python.org/downloads/release/python-3119/)  
- **Git** *(optional, for cloning the repository)* – [Download](https://git-scm.com/downloads)

> **Why Python 3.11.9?**  
> Some ML dependencies (e.g., XGBoost, SHAP) are more stable on this version. The fixes made here *may* allow newer versions, but they are not guaranteed to work.

---

## 3. Download & Open the Project

**Download ZIP**
1. Download the `.zip` file of the repository.
2. Extract it to any location (Desktop, Documents, etc.).
3. Open **VS Code** → `File` → `Open Folder…` → select the extracted folder `BA-main`.

---

## 4. Running the Project

After downloading and opening the project in VS Code, the workflow can be executed either fully inside the main Jupyter Notebook or by running the individual scripts.
The pipeline is modular, so each step can be run independently, but for a first-time reproduction it is recommended to run the full process.


## **Step 1 – Environment Setup**

Creating a virtual environment ensures that all dependencies are installed in an isolated space and do not interfere with other Python projects on your system.

1. Check if **Python 3.11.9** is installed correctly:  
   ```bash
   python -V
   ```
If not detected, repair the installation and restart the machine. See **6. Troubleshooting** for more information.

2. In VS Code, click **Select Kernel**, choose **Python Environments...** and pick your **global Python 3.11.9** installation for now.
   
> In VS Code, this option is located either in the **upper right** or **bottom-left** corner.

3. **(Windows only)** Adjust PowerShell Execution Policy

Powershell prevents the use of scripts by default. To bypass this issue for the duration of this session, use the following command:
   ```bash
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   ```
4. Creating the virtual environment
   
Use the following command to create a new virtual environment. 
> Name for the environment (second venv in this case) can be chosen at will!

This guide assumes you use venv as your actual environment name:
   ```bash
   python -m venv venv
   ```
This creates a new folder venv/ that contains the isolated Python environment.

5. Activate the virtual environment
   
   Windows (PowerShell):
   ```bash
   .\venv\Scripts\activate
   ``` 
   macOS/Linux (bash/zsh):
   ```bash
   source venv/bin/activate
   ```
If not done automatically, please re-select your Kernel as your new virtual environment.

This ensures that VS Code uses the same environment where you will install your dependencies.

6. Install dependencies:   
   ```bash
   pip install -r requirements.txt
   ```
This ensures that the exact dependency versions from the thesis environment are installed.


## **Step 2 – Add Reddit API Credentials**

To scrape new Reddit data, you must provide your own Reddit API credentials. 
1. Create a new .env file in the project root
   
> Right-click on BA in VS Code’s Explorer panel → New File → name it .env.

2. Add your Reddit API credentials:
   ```
   CLIENT_ID=your_client_id
   CLIENT_SECRET=your_client_secret
   USER_AGENT=your_user_agent
   ```
   - Into .env with *your_xxx* as **your actual Reddit-API** credentials.
     
  >  Obtain credentials: Reddit-API [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)

  > These keys can be generated in your Reddit account under Preferences → Apps → Create App → Select script.

  > Enter any name for the API, description and link info can be ignored.

  > Use http://localhost:65010/authorize_callback for redirection, this is mandatory.

Without the .env-file, the pipeline will only load existing data from the database and skip scraping.


**Step 3 – (Optional) Reset for a Clean Run**

Delete BA/data/processed/reddit_dota2_analysis.db (to force re-scraping) and optionally clear the BA/reports/ folder (to force regeneration of all plots). 

Subfolders will be automatically recreated when the scripts run.

For a full, clean reproduction:

Windows:
   ```bash
   del BA\data\processed\reddit_dota2_analysis.db
   ```
macOS/Linux:
   ```bash
   rm BA/data/processed/reddit_dota2_analysis.db
   ```
  - Optionally delete all subfolders of BA/reports/ → Forces regeneration of plots
  - Optionally delete all subfolders of BA/logs/ → Clears previous logs

Subfolders are automatically recreated by the scripts.


**Step 4 – (Recommended) Ensure Unicode Output in Terminal**

To prevent encoding errors for special characters (e.g., emojis) in logs:

Windows:
   ```bash
   $env:PYTHONIOENCODING="utf-8"
   ```
macOS/Linux: Usually not required.


**Step 5 – Execute the Workflow**

Option A – Full Workflow in the Notebook (data_pipeline & eda_stats):
   1. Open notebooks/1_data_extraction/reddit_scraper.ipynb
   2. Restart the kernel everytime before running cells
   3. Use Run All to execute all cells in order

> This runs **data preparation, EDA, and statistical tests** automatically, logs are saved in BA/logs/

> Markdown cells in the notebook will also expand when command Run All is used. Double click to de-size Markdown cells.

**Note:** Machine Learning must be run separately (see Option B).

Option B – Run Individual Pipeline Steps:

If you wish to run only certain parts of the pipeline:

1. **Data preparation**:
   ```bash
   python BA/src/data/prepare_data.py
   ```
2. **EDA & statistical tests**: Run the notebook after data preparation is complete.
   
3. **Machine learning**:
   ```bash
   python BA/src/models/train_model.py
   ```

---

## **5. Viewing Results**

After successfully running all workflow steps, you can review the generated outputs:

1. **Data Outputs**
   - Location: `BA/data/processed/reddit_dota2_analysis.db`
   - This SQLite database contains all processed Reddit comments with engineered features.
   - You can inspect it using:
     - SQLite VS Code extension
     - External tools such as DB Browser for SQLite

2. **Visualizations**
   - Location: `BA/reports/figures/`
   - Includes:
     - EDA plots
     - Final ML plots (feature importance, SHAP)
   - Subfolders are created automatically if missing.

3. **Machine Learning Artifacts**
   - Location: `BA/models/`
   - Contains:
     - Trained model files
     - Preprocessing pipeline objects
   - These can be reloaded for inference or further analysis.

4. **Log Files**
   - Location: `BA/logs/`
   - Files:
     - `data_pipeline.log` – logs from data preparation
     - `eda_stats.log` – logs from EDA and statistical tests
     - `ml_training.log` – logs from ML pipeline
   - Use these to trace execution order and debug issues.

---

## **6. Troubleshooting**

If you encounter issues during setup or execution, check the following:

1. **Python Not Found in VS Code**
   - Ensure Python 3.11.9 is installed and visible in VS Code’s kernel list.
   - Check by running command:
       ```bash
       python -V
       ``` 
   - If missing:
     - Run the Python installer
     - Select **Repair**
     - Restart your computer (automatically asked when repaired successfully)

2. **VS Code or Python Not in PATH (Windows Only)**
   - Open *Environment Variables* in Windows
   - Edit the `PATH` variable
   - Ensure both Python installation directory and VS Code path are listed

3. **No Data Scraped**
   - Verify `.env` file contains **valid Reddit API credentials**
   - Delete `reddit_dota2_analysis.db` to force re-scraping

4. **Plots Not Generated**
   - Ensure the `reports/` directory is not write-protected
   - Empty the folder before running the pipeline to confirm regeneration

5. **Unicode Errors in Terminal**
   - Set encoding in the current terminal session:
     
     - **Windows PowerShell**:
       ```powershell
       $env:PYTHONIOENCODING="utf-8"
       ```
     - **macOS/Linux**: Usually not required
Purely cosmetic, scraping Tundra/TI11 for data will result in a trace back error due to an emoji but the Code will still continue to run.
> **Tip:** Always check the corresponding `.log` files for detailed error messages.

## **7. Optional Steps**

These steps are not strictly required for running the workflow but can help with verification, reproducibility, and customization.

1. **Force Complete Workflow Verification**
   - Delete the following before running the pipeline:
     - `BA/data/processed/reddit_dota2_analysis.db` → Forces full data scraping
     - Entire `BA/reports/` directory → Forces regeneration of all plots
   - Subfolders will be automatically recreated when the scripts run.

2. **Change Analysis Scope**
   - Modify `config/teams.json` or `config/keywords.json` to track different teams, players, or keywords.
   - Update `config.py` for:
     - Event date ranges
     - Feature selection
     - API limits

3. **Re-run Machine Learning with New Parameters**
   - Adjust hyperparameters in `config.py` for:
     - TF-IDF vectorization
     - Model configurations (Linear Regression, XGBoost)
   - This is useful for testing different modeling strategies.

---

## **8. Project Structure Overview**

Below is the folder structure for quick navigation:

BA/

├── config/ # Configuration files (.py, .json)

│ ├── keywords.json # Keywords for detection

│ └── teams.json # Team/player names

│

├── data/

│ ├── processed/ # Processed SQLite database

│ └── raw/ # Optional raw data storage

│
├── logs/ # Execution logs

│ ├── data_pipeline.log # Data extraction & processing logs

│ ├── eda_stats.log # EDA & statistical tests logs

│ └── ml_training.log # Machine learning logs

│

├── models/ # Trained models and preprocessing objects

│

├── notebooks/ # Jupyter notebooks

│ └── 1_data_extraction/

│ └── reddit_scraper.ipynb

│

├── reports/

│ ├── figures/ # All generated plots and visualizations

│ │ ├── final_plots/ # Final 10 chosen Plots for Bachelor Thesis

│ │ ├── eda/ # EDA plots

│ │ ├── statistical_tests/ # Statistical test plots

│ │ ├── ml_plots/ # Machine learning plots

│ │ └── pearson_correlation/ # Pearson Correlation Matrix

│

├── src/

│ ├── data/ # Data extraction and preparation scripts

│ ├── features/ # Feature engineering and text processing

│ ├── models/ # Training and evaluation scripts

│ ├── utils/ # Utility functions (e.g., logging)

│ └── visualizations/ # Plot generation scripts

├── .env # Personal Reddit-API credentials (not Shown)

├── .gitignore # Prevents files to be uploaded to GitHub (.env, also not shown)

├── config.py # Central porject configuration

└── requirements.txt # Python dependencies

---

## **9. License & Attribution**

This project is developed as part of a **Bachelor’s Thesis** at the *Hochschule für Wirtschaft und Recht Berlin (HWR Berlin)*.

**Author:** Duy Viet Kuschy  
**Supervisor:** Prof. Dr. Diana Hristova

**Assistant Supervisor:** Prof. Dr. Alexander Eck

- The code and documentation are provided **for academic and research purposes only**.
- Redistribution, modification, or commercial use is **not permitted** without explicit permission from the author.
- All third-party libraries used are licensed under their respective open-source licenses.
- The dataset created by this project is partially sourced from publicly available Reddit comments via the Reddit API, subject to Reddit’s [API Terms of Use](https://www.redditinc.com/policies/data-api-terms).

If you use or reference this work, please cite it appropriately:

Kuschy, D. V. (2025). Online Engagement with Organizational News on Reddit: An Analysis Using the Example of E-Sports Player Transfers and the Online Presence of Organizations in Rankings and Tournaments. Bachelor's Thesis, Hochschule für Wirtschaft und Recht Berlin.

---

## **10. Final Notes**

- This repository contains the **finalized version** of the codebase used for the Bachelor's Thesis.
- All scripts, configurations, and modules have been tested for reproducibility on a clean setup using **Python 3.11.9** and the exact package versions specified in `requirements.txt`.
- While the pipeline is modular and flexible enough to adapt to new datasets or analysis contexts, any modifications should be done carefully to avoid breaking dependencies between modules.
- The **primary focus** of the code is:
  1. **Reproducibility** – consistent results across different systems.
     
  3. **Modularity** – clear separation of data collection, processing, analysis, and visualization.
     
  5. **Transparency** – detailed logging in `logs/` and well-documented functions for clarity.

If you encounter issues or wish to adapt the code for other research purposes, please review the **Troubleshooting & Known Issues** section before making changes.

---
