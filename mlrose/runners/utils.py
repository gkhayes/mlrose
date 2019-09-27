import os


def build_data_filename(output_directory, runner_name, experiment_name, df_name, ext=''):
    # ensure directory exists
    try:
        os.makedirs(os.path.join(output_directory, experiment_name))
    except:
        pass

    # return full filename
    if len(ext) > 0 and not ext[0] == '.':
        ext = f'.{ext}'
    return os.path.join(output_directory,
                        experiment_name,
                        f'{runner_name.lower()}__{experiment_name}__{df_name}{ext}')