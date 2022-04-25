from setuptools import setup, find_packages

setup(
    name='tevatron',
    version='0.0.1',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    url='https://github.com/ielab/CharacterBERT-DR',
    license='Apache 2.0',
    author='Luyu Gao, Shengyao Zhuang',
    author_email='luyug@cs.cmu.edu, s.zhuang@uq.edu.au',
    description='Customized Tevatron that designed for CharacterBERT-based DR.',
    python_requires='>=3.7',
    install_requires=[
        "transformers>=4.3.0,<=4.9.2",
        "datasets>=1.1.3"
    ]
)
