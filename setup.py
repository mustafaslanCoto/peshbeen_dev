from setuptools import setup, find_packages

VERSION = "0.0.4"
DESCRIPTION = "Auto regressive bagging and boosting package"
LONG_DESCRIPTION = "Forecasting using historical data as predictors"

setup(
    name="peshbeen",
    version=VERSION,
    author="Mustafa Aslan",
    author_email="mustafaslan63@email.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    url="https://github.com/mustafaslanCoto/peshbeen_dev",
    install_requires=[
        "setuptools>=68",   # needed for pkg_resources (used by hyperopt)
        "numpy>=2.0",
        "pandas>=2.2",
        "scikit-learn>=1.5",
        "scipy>=1.15,<1.16",
        "statsmodels>=0.14",
        "matplotlib>=3.9",
        "seaborn>=0.13",
        "hyperopt>=0.2.7",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "catboost>=1.2",
        "cubist>=1.0.0",
        "statsforecast>=1.7",
        "window-ops>=0.0.15",
        "numba>=0.60",
        "great-tables>=0.10",
    ],
    python_requires=">=3.10,<3.14",
    keywords=["python", "peshbeen", "forecasting", "time series", "machine learning"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)