#!/usr/bin/env python3
import codecs
import os
from distutils.core import setup

def find_version(*file_paths):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, *file_paths), 'r') as fp:
        for line in fp:
            if line.startswith('__version__'):
                return line[line.index('=')+1:].replace("'",'').strip()
    raise RuntimeError("Unable to find version string.")

__version__ = find_version('dahua_devices/__init__.py')
GITHUB_VERSION = 'v' + __version__
setup(
    name='dahua-devices',
    packages=['dahua_devices'],
    version=__version__,
    license='MIT',
    description='Python Dahua API bindings',
    author='Alexander Ryazanov',
    author_email='alryaz@xavux.com',
    url='https://github.com/alryaz/dahua-devices-python',
    download_url='https://github.com/alryaz/dahua-devices-python/archive/' + GITHUB_VERSION + '.tar.gz',
    keywords=['Dahua', 'API', 'CCTV'],
    install_requires=['aiohttp',],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
