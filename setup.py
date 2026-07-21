from setuptools import setup, find_packages

setup(
    name="fashion-image-text-alignment",
    version="1.0.0",
    author="Vinay M Madgi", "Sahana Gidnandi", "Nisha D", "Kshitij H", "Channabasappa Muttal",
    author_email="vinaymadgi28@gmail.com", "sahanagidnandi@gmail.com", "nishadodwad5@gmail.com", "01fe23bci010@kletech.ac.in", "channabasappa.muttal@kletech.ac.in",
    description=(
        "Official implementation of the Springer paper 'Vision-Language Models for Fashion Conversational Assistants with Multimodal Dialogues'"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Vinay3579-code/Image-Text-Alignment-Framework",
    license="MIT",

    packages=find_packages(),

    include_package_data=True,

    python_requires=">=3.10",

    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "open-clip-torch>=2.24.0",
        "transformers>=4.40.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "Pillow>=10.2.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.4.0",
    ],

    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "pytest",
        ]
    },

    keywords=[
        "vision-language",
        "fashion",
        "clip",
        "eva02",
        "image-text alignment",
        "multimodal",
        "deep learning",
        "pytorch",
    ],

    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
