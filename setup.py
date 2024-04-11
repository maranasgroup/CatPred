from setuptools import find_packages, setup

__version__ = "0.0.1"

# Load README
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="catpred",
    author="Veda Sheersh Boorla, Costas D. Maranas",
    author_email="mailforveda@gmail.com",
    description="A comprehensive framework for deep learning in vitro enzyme kinetic parameters kcat, Km and Ki",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maranasgroup/catpred",
    download_url=f"https://github.com/maranasgroup/catpred/v_{__version__}.tar.gz",
    project_urls={
        "Documentation": "https://github.com/maranasgroup/catpred/",
        "Source": "https://github.com/maranasgroup/catpred",
        "PyPi": "",
        "Demo": "https://tiny.cc/catpred",
    },
    license="MIT",
    packages=find_packages(),
    package_data={"catpred": ["py.typed"]},
    entry_points={
        "console_scripts": [
            "catpred_train=catpred.train:catpred_train",
            "catpred_predict=catpred.train:catpred_predict",
        ]
    },
    install_requires=[
        "matplotlib>=3.1.3",
        "numpy>=1.18.1",
        "pandas>=1.0.3",
        "pandas-flavor>=0.2.0",
        "scikit-learn>=0.22.2.post1",
        "tensorboardX>=2.0",
        "sphinx>=3.1.2",
        "torch>=1.4.0",
        "tqdm>=4.45.0",
        "typed-argument-parser>=1.6.1",
        "rdkit>=2020.03.1.0",
        "scipy<1.11 ; python_version=='3.7'",
        "scipy>=1.9 ; python_version=='3.8'",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "bioinformatics",
        "machine learning",
        "enzyme function prediction",
        "message passing neural network",
    ],
)
