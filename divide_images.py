import os 
import shutil


def test_train(src_folder, test_destination, train_destination):
    sub_folders = ['with_mask', 'without_mask']
    # specify the condition to separate files
    condition = lambda filename: filename.endswith("5.jpg")
    
    sanity_path = './sanity_data'

    if not os.path.exists(sanity_path):
        os.makedirs(sanity_path)
    
    for subfolder in sub_folders:
        subfolder_path = os.path.join(sanity_path, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    if not os.path.exists(test_destination):
        os.makedirs(test_destination)
    
    for subfolder in sub_folders:
        subfolder_path = os.path.join(test_destination, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    if not os.path.exists(train_destination):
        os.makedirs(train_destination)
    
    for subfolder in sub_folders:
        subfolder_path = os.path.join(train_destination, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    
    for folder in os.listdir(src_folder):
        i = 0
        for filename in os.listdir(os.path.join(src_folder, folder)):
            src_file = os.path.join(src_folder, folder, filename)

            # check if the file meets the condition
            if condition(filename):
                dest_file = os.path.join(test_destination, folder ,filename)
            else:
                dest_file = os.path.join(train_destination, folder, filename)

            # move the file to the appropriate destination folder
            shutil.copy(src_file, dest_file)
            if i < 25:
                sanity_file = os.path.join(sanity_path, folder, filename)
                shutil.copy(src_file, sanity_file)
            i += 1
