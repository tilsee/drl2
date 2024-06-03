import os


def save_run_info(config: dict[str, dict], seed: int, algorithm_name: str):
    conf_path = config['Common']['configuration_save_path']
    if not os.path.exists(conf_path):
        os.makedirs(conf_path)

    LINE_SEPARATOR = '-----------------\n'
    with open(conf_path + 'run_configuration.txt', 'w') as f:
        f.write('Run Configuration\n')
        f.write(LINE_SEPARATOR)
        f.write(f"Experiment name: {config['Common']['experiment_name']}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Algorithm: {algorithm_name}\n")
        f.write(LINE_SEPARATOR)
        f.write('Arguments\n')
        for k, v in config._sections.items():
            if isinstance(v, dict):
                f.write(LINE_SEPARATOR)
                f.write(f'{k}\n')
                for i in v.items():
                    f.write(f'\t{i}\n')
                f.write(LINE_SEPARATOR)
        f.write('\n')
        f.write('')
