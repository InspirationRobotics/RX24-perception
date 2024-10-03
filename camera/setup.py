from setuptools import setup, find_packages

setup(
    name='camera_core',
    version='0.1',
    description='Core camera package for Inspiration RobotX',
    author='Eesh Vij',
    packages=find_packages(include=['camera_core'])
)

'''
Required packages:
pyusb

'''