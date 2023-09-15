from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='mosaic',
      version='1.0',
      packages=find_packages(),
      install_requires=required,
      author='Gyan Tatiya',
      author_email='Gyan.Tatiya@tufts.edu',
      description='MOSAIC (Multi-modal Object property learning with Self-Attention and Integrated Comprehension)',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/gtatiya/MOSAIC',
      license='MIT License',
      python_requires='>=3.10',
      )
