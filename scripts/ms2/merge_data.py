import os
import h5py

# Function to merge HDF5 files
def merge_h5_files(input_dir, output_file):
    # Create an empty HDF5 file to store the merged data
    # with h5py.File(output_file, "w") as output_h5:
    input_paths = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".h5") and filename != "trajectory.h5":
            file_path = os.path.join(input_dir, filename)
            input_paths.append(file_path)
    from mani_skill2.trajectory.merge_trajectory import merge_h5
    merge_h5(output_file, input_paths, recompute_id=True)

if __name__ == "__main__":
    input_directory = "demos/v0/rigid_body/TurnFaucet-v0/"
    output_file = "demos/v0/rigid_body/TurnFaucet-v0/trajectory.h5"

    merge_h5_files(input_directory, output_file)