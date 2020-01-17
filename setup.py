import setuptools

setuptools.setup(
	name='chess_gym',
	version='0.1',
	description='A chess environment for AI with the gym interface',
	packages=setuptools.find_packages(),
	install_requires=['gym', 'numpy', 'opencv-python'],
	include_package_data=True
)
