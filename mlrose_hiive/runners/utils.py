import os


def build_data_filename(output_directory, runner_name, experiment_name, df_name, x_param='', y_param='', ext=''):
    # ensure directory exists
    try:
        os.makedirs(os.path.join(output_directory, experiment_name))
    except:
        pass

    # return full filename
    if len(ext) > 0 and not ext[0] == '.':
        ext = f'.{ext}'
    if len(x_param) > 0 and not x_param[0] == '_' and not x_param[-1] == '_':
        x_param = f'_{x_param}_'
    if len(y_param) > 0 and not y_param[0] == '_' and not y_param[-1] == '_':
        y_param = f'_{y_param}'
    return os.path.join(output_directory,
                        experiment_name,
                        f'{runner_name.lower()}__{experiment_name}__{df_name}{x_param}{y_param}{ext}')