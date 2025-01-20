from setuptools import setup, find_packages

setup(
    name="hoomanai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.0.0",
        "typing-extensions>=4.0.0",
        "python-dotenv>=0.19.0",
        "openai>=1.0.0",
        "langchain>=0.1.0",
        "uuid>=1.30",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An extensible framework for building and orchestrating AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/hoomanai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)