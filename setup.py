from setuptools import setup, find_packages # type: ignore

setup(
    name='sales_prediction',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter',
    ],
    entry_points={
        'console_scripts': [
            'data_preprocessing=src.data_preprocessing:main',
            'feature_engineering=src.feature_engineering:main',
            'model_training=src.model_training:main',
            'model_evaluation=src.model_evaluation:main',
        ],
    },
)
