from setuptools import setup, find_packages

setup(
    name='my_python_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
    ],
    entry_points={
        'console_scripts': [
            'my_python_project=my_python_project.main:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_python_project',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
