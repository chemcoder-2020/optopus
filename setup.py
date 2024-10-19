from setuptools import setup, find_packages

setup(
    name='optopus',
    version='0.1.1',
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
)
