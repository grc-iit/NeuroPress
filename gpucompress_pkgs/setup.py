from setuptools import setup, find_packages

setup(
    name='gpucompress_pkgs',
    version='1.0.0',
    description='Jarvis-CD packages for GPUCompress simulation deployments',
    packages=find_packages(),
    install_requires=[
        'jarvis-cd',
    ],
    package_data={
        '': ['*.yaml', '*.yml', '*.md'],
    },
)
