from setuptools import setup, find_packages

setup(
    name="motndp",
    description="Multi-Objective Transport Network Design Problem using the city of Amsterdam",
    version="0.0.1",
    install_requires=["gymnasium==0.27.1"],
    packages=find_packages(),
    include_package_data=True,
)