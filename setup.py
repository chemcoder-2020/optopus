from setuptools import setup, find_packages

setup(
    name='optopus',
    version='0.2.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    author='Huy Nguyen',
    author_email='huynguyen2406@gmail.com',
    description='A Python project for options trading and backtesting.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/optopus',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        "joblib>=1.2.0",
        "loguru>=0.6.0",
        "matplotlib>=3.5.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "setuptools>=58.0.0",
        "tqdm>=4.62.0"
    ],
)
