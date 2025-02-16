from setuptools import setup, find_packages

setup(
    name='optopus',
    version='0.9.4-dev3',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={
        "optopus": ["templates/*/*"],
    },
    author='Huy Nguyen',
    author_email='huynguyen2406@gmail.com',
    description='A Python project for options trading and backtesting.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/chemcoder-2020/optopus',
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
        "httpx>=0.23.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "setuptools>=58.0.0",
        "tqdm>=4.62.0",
        "dill>=0.3.4",
        "configparser>=5.3.0",
        "statsforecast>=1.0.0",
        "neuralforecast>=1.0.0",
        "sktime>=0.3.0",
        "plotly>=5.0.0",
    ],
    entry_points={
        'console_scripts': [
            'setup-optopus-backtest=optopus.cli.setup_backtest:main',
            'bot-status=optopus.cli.bot_status:main',
        ],
    },
)
