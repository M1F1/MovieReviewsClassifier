import os

__file_root__ = 'C:\\Users\\Qbit\\Inzynierka'

def move_to_main_location():
	os.chdir(__file_root__)

def move_to_data_location():
	os.chdir(os.path.join(__file_root__, 'Data'))

def move_to_model_location():
	os.chdir(os.path.join(__file_root__, 'Models'))

def move_to_plot_location():
	path_to_plot_location =  os.path.join(__file_root__, 'Plots')
	if not os.path.exists(path_to_plot_location):
		os.mkdir(path_to_plot_location)
	os.chdir(path_to_plot_location)

def list_directory():
	filenames = os.listdir()
	numerated_filenames = list(zip(filenames, range(len(filenames))))
	for nf in numerated_filenames:
		print(nf)
	return filenames

