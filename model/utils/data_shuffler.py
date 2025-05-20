import os
import random
import shutil

class DataShuffler:
    def __init__(self,
                 directory_with_data: str,
                 directory_with_new_data: str,
                 train_ratio: float = 0.9,
                 validation_ratio: float = 0.05,
                 random_seed: int = 42  
                 ):
        self.__source_folder = os.path.abspath(directory_with_data)
        self.__output_folder = os.path.abspath(directory_with_new_data)

        self.__train_ratio = train_ratio
        self.__val_ratio = validation_ratio
        self.__test_ratio = 1 - self.__train_ratio - self.__val_ratio
        self.__random_seed = random_seed

        self.__rng = random.Random()
        if self.__random_seed is not None:
            self.__rng.seed(self.__random_seed)

        if self.__train_ratio + self.__val_ratio + self.__test_ratio != 1:
            raise ValueError("Sum of split ratios should be equal to 1.")
        if not (0 < self.__train_ratio < 1 and 0 < self.__train_ratio < 1
                and 0 < self.__train_ratio < 1):
             raise ValueError("Invalid values for ratios.")
        self.__execute()

    def __execute(self):
        print(f"\nStarting data splitting process...")

        if not os.path.exists(self.__source_folder):
            print(f"Error: Source folder '{self.__source_folder}' does not exist.")
            return
        
        train_dir = os.path.join(self.__output_folder, 'train')
        val_dir = os.path.join(self.__output_folder, 'val')
        test_dir = os.path.join(self.__output_folder, 'test')

        try:
            os.makedirs(self.__output_folder, exist_ok=True)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            print(f"Created base output directories: '{train_dir}', \
                  '{val_dir}', and '{test_dir}'")
        except OSError as e:
            print(f"Error creating directories: {e}")
            return
        
        for class_name in sorted(os.listdir(self.__source_folder)):
            source_class_path = os.path.join(self.__source_folder, \
                                              class_name)
            if os.path.isdir(source_class_path):
                print(f"\nProcessing class: {class_name}")
                train_class_path = os.path.join(train_dir, class_name)
                val_class_path = os.path.join(val_dir, class_name)
                test_class_path = os.path.join(test_dir, class_name)
                os.makedirs(train_class_path, exist_ok=True)
                os.makedirs(val_class_path, exist_ok=True)
                os.makedirs(test_class_path, exist_ok=True)

                try:
                    all_files = [
                        f for f in os.listdir(source_class_path)
                        if os.path.isfile(os.path.join(source_class_path, f))
                    ]
                except OSError as e:
                    print(f"Warning: Could not read files in {source_class_path}. Skipping. Error: {e}")
                    continue

                if not all_files:
                    print(f"Warning: No files found in {source_class_path}. Skipping.")
                    continue

                self.__rng.shuffle(all_files)
                n_total = len(all_files)
                
                train_idx_end = round(n_total * self.__train_ratio)
                val_idx_end = round(n_total * (self.__train_ratio + self.__val_ratio))

                train_idx_end = min(train_idx_end, n_total)
                val_idx_end = min(val_idx_end, n_total)
                val_idx_end = max(val_idx_end, train_idx_end)

                train_files = all_files[:train_idx_end]
                val_files = all_files[train_idx_end:val_idx_end]
                test_files = all_files[val_idx_end:]

                print(f"  Total files: {n_total}")
                print(f"    Train files: {len(train_files)}")
                print(f"    Validation files: {len(val_files)}")
                print(f"    Test files: {len(test_files)}")

                def __copy_files(files_to_copy, destination_path, set_name):
                    print(f"  Copying {len(files_to_copy)} {set_name} files to {destination_path}...")
                    copied_count = 0
                    for file_name in files_to_copy:
                        source_file_path = os.path.join(source_class_path, file_name)
                        dest_file_path = os.path.join(destination_path, file_name)
                        try:
                            shutil.copy2(source_file_path, dest_file_path)
                            copied_count += 1
                        except Exception as e:
                            print(f"Error copying {source_file_path} to {dest_file_path}: {e}")
                    if copied_count != len(files_to_copy):
                        print(f"Warning: Only {copied_count} out of {len(files_to_copy)} {set_name} files were copied successfully for class {class_name}.")

                __copy_files(train_files, train_class_path, "training")
                __copy_files(val_files, val_class_path, "validation")
                __copy_files(test_files, test_class_path, "test")
            else:
                print(f"Skipping '{class_name}' as it is not a directory.")
        print("\nData splitting complete!")