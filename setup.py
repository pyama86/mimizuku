from setuptools import find_packages, setup

setup(
    name="mimizuku",
    version="0.2.28",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "joblib",
        "orjson",
    ],
    entry_points={
        "console_scripts": [
            "mimizuku=mimizuku.model:main",  # コマンドラインで実行する場合
        ],
    },
    author="pyama86",
    author_email="www.kazu.com@gmail.com",
    description="A package for anomaly detection using Isolation Forest for Wazuh Alerts",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pyama86/mimizuku",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
