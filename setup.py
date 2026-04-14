from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="fashion-feature-extraction",
    version="0.1.0",
    description="Pipeline de extraccion de atributos visuales de prendas de moda",
    author="Damian Ilkow",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fashion-extract=src.pipeline:main",
        ],
    },
)
