from setuptools import setup, find_packages

setup(
    name="spectrum",
    version="0.1.0",
    description="Spectrum analysis for neural network layers",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "tqdm",
        "prompt_toolkit",
    ],
    python_requires=">=3.7",
)
