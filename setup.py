from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="physics_informed_neural_network",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Physics-Informed Neural Network for Power Grid and Renewable Energy Applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/physics_informed_neural_network",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "black>=21.5b2",
            "isort>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pinn-train=src.cli.train:main",
            "pinn-predict=src.cli.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "physics_informed_neural_network": [
            "config/*.yaml",
            "models/*.h5",
        ],
    },
) 