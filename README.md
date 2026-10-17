# ML Tool for Post-Heated FRP-Repaired Concrete

Public Streamlit interface converted from the supplied working PySimpleGUI code. Predicts ultimate compressive strength (`fcc,T`), strength enhancement ratio (`fcc,T / fco,T`), and failure mode (Rupture, Debonding, Mixed).

## Install and run (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Before running, put **all seven** original trusted `.joblib` files in `Results/Models/`, using the same filenames as `app.py`.

## Deploy

1. Create a **public** GitHub repository and upload `app.py`, `requirements.txt`, `.streamlit/config.toml`, `README.md`, `.gitignore`, and the `Results/Models/` folder containing seven files.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Create an app from the GitHub repository. Set branch to `main` and entrypoint to `app.py`.
4. In Advanced settings, select **Python 3.11** (or the version used to create and test your models).
5. Deploy, test a known specimen, and share your `https://your-app-name.streamlit.app` URL.

The initial requirements use broad compatible version ranges, not guarantees of model serialization compatibility. If loading fails, replicate the **exact training versions** of Python, scikit-learn, XGBoost, NumPy and SciPy in the cloud. Models serialized via joblib may break across versions. Never load untrusted joblib files.

## Validation checklist

- Verify that `FEATURE_COLUMNS` matches the exact feature order used during training. The supplied PySimpleGUI tool constructs the raw array in the order of its `numerical_input_ranges` followed by `Cycle`, `Cooling`; that order was retained.
- Compare predictions on at least 5 known specimen inputs with your original desktop application before sharing the web URL.
- Ensure all units and category encodings match your training set.
- Review whether publishing serialized model artifacts under a public license is intended; public GitHub makes them downloadable.
- This application does **not** claim calibrated prediction intervals. Individual RF tree quantiles in the desktop source do not establish 95% empirical coverage.
- Free hosting may sleep and imposes memory/resource limits. Test deployment size and performance.
- Session data is transient; downloading Excel retains a local copy.

## Planned website integration

Add a link from your research website's Tools/Software page to the Streamlit URL. Later, you can embed the app using an iframe if the hosting platform and security headers permit it.

© 2026 I. A. Tijani, M. Elshorbagi, & C. Jiang. All rights reserved.
