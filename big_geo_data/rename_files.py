
'''Prunes the '.' or "Period" symbol from the file names in the given directory'''

import os
import pandas as pd

def rename_files():
    folders_path = "./Sample800"
    for directname, directnames, files in os.walk(folders_path):
        for f in files:
            # Split the file into the filename and the extension, saving
            # as separate variables
            filename, ext = os.path.splitext(f)
            if "." in filename:
                # If a '.' is in the name, rename, appending the suffix
                # to the new file
                new_name = filename.replace(".", "_")
                os.rename(
                    os.path.join(directname, f),
                    os.path.join(directname, new_name + ext))


    # Do the same thing with the file names in the coordinate_in_Bbox csv so that they match

    def remove_period(filename):
        name, ext = os.path.splitext(filename)
        if "." in name:
            name = name.replace(".", "_")
            filename = name + ext
        return filename

    df = pd.read_csv('Coordinate_in_Bbox.csv')
    df['Sample Name'] = df['Sample Name'].apply(remove_period)
    print(df)
    df.to_csv('./coordinates_Bbox.csv')

