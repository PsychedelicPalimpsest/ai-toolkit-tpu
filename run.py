import os
import sys
from dotenv import load_dotenv
# Load the .env file if it exists
load_dotenv()
os.environ["HF_XET_HIGH_PERFORMANCE"] = os.getenv("HF_XET_HIGH_PERFORMANCE", "1")
os.environ["HF_HUB_DISABLE_XET"] = os.getenv("HF_HUB_DISABLE_XET", "0")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
seed = None
if "SEED" in os.environ:
    try:
        seed = int(os.environ["SEED"])
    except ValueError:
        print(f"Invalid SEED value: {os.environ['SEED']}. SEED must be an integer.")

sys.path.insert(0, os.getcwd())

# The UI launches jobs with no console; keep anything we shell out to (torch
# compiles, HF git downloads) from flashing a console window. Must come before
# any import that might spawn a subprocess.
from toolkit.win_console import suppress_child_consoles
suppress_child_consoles()

# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# set torch to trace mode
import torch
    
# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    torch.autograd.set_detect_anomaly(True)

if seed is not None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        from toolkit.xla_utils import seed_all
        seed_all(seed)
    except Exception:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

import argparse
from toolkit.job import get_job
from toolkit.accelerator import get_accelerator
from toolkit.print import print_acc, setup_log_to_file

accelerator = get_accelerator()


def print_end_message(jobs_completed, jobs_failed):
    if not accelerator.is_main_process:
        return
    try:
        from toolkit.xla_utils import is_master_ordinal
        if not is_master_ordinal():
            return
    except Exception:
        pass
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print_acc("")
    print_acc("========================================")
    print_acc("Result:")
    if len(completed_string) > 0:
        print_acc(f" - {completed_string}")
    if len(failure_string) > 0:
        print_acc(f" - {failure_string}")
    print_acc("========================================")


def _find_config_path(config_file):
    # Mirror toolkit/config.py resolution without importing the job stack.
    try:
        from toolkit.paths import TOOLKIT_ROOT
    except Exception:
        TOOLKIT_ROOT = os.getcwd()
    candidate = os.path.join(TOOLKIT_ROOT, 'config', config_file)
    if os.path.exists(candidate):
        return candidate
    for ext in ('.json', '.jsonc', '.yaml', '.yml'):
        if os.path.exists(candidate + ext):
            return candidate + ext
    if os.path.exists(config_file):
        return config_file
    abs_path = os.path.join(os.getcwd(), config_file)
    if os.path.exists(abs_path):
        return abs_path
    return None


def _resolve_tpu_cores(config_file_list, cli_cores=None):
    """How many TPU cores to spawn. Priority: --tpu_cores > AITK_TPU_CORES
    env > max(train.tpu_num_cores) over the config files. Always >= 1."""
    if cli_cores is not None:
        try:
            if int(cli_cores) >= 1:
                return int(cli_cores)
        except Exception:
            pass
    env_cores = os.environ.get('AITK_TPU_CORES', None)
    if env_cores is not None:
        try:
            if int(env_cores) >= 1:
                return int(env_cores)
        except Exception:
            pass
    cores = 1
    for config_file in config_file_list:
        try:
            path = _find_config_path(config_file)
            if path is None:
                continue
            with open(path, 'r', encoding='utf-8') as f:
                if path.endswith('.json') or path.endswith('.jsonc'):
                    import json
                    data = json.load(f)
                else:
                    import yaml
                    data = yaml.safe_load(f)
            processes = ((data or {}).get('config') or {}).get('process') or []
            for proc in processes:
                try:
                    v = ((proc or {}).get('train') or {}).get('tpu_num_cores', 1)
                    cores = max(cores, int(v or 1))
                except Exception:
                    continue
        except Exception:
            continue
    return max(1, cores)


def _tpu_worker(index, config_file_list, args):
    """xmp.spawn target: one process per TPU core.

    The module-level ``accelerator`` below was created at import time, before
    the spawn machinery assigned this process its core. Drop it so every
    downstream ``get_accelerator()`` binds this worker's own XLA device.
    """
    try:
        import toolkit.accelerator as _acc_mod
        _acc_mod.global_accelerator = None
    except Exception:
        pass
    global accelerator
    try:
        accelerator = get_accelerator()
    except Exception:
        pass
    _run_configs(config_file_list, args)


def _run_configs(config_file_list, args):
    if args.log is not None:
        setup_log_to_file(args.log)

    if len(config_file_list) == 0:
        raise Exception("You must provide at least one config file")

    jobs_completed = 0
    jobs_failed = 0

    if accelerator.is_main_process:
        print_acc(f"Running {len(config_file_list)} job{'' if len(config_file_list) == 1 else 's'}")

    for config_file in config_file_list:
        try:
            job = get_job(config_file, args.name)
            job.run()
            job.cleanup()
            jobs_completed += 1
        except Exception as e:
            print_acc(f"Error running job: {e}")
            jobs_failed += 1
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_end_message(jobs_completed, jobs_failed)
                raise e
        except KeyboardInterrupt as e:
            try:
                job.process[0].on_error(e)
            except Exception as e2:
                print_acc(f"Error running on_error: {e2}")
            if not args.recover:
                print_acc("")
                print_acc("========================================")
                print_acc("Job stopped")
                print_acc("========================================")
                sys.exit(0)


def main():
    parser = argparse.ArgumentParser()

    # require at lease one config file
    parser.add_argument(
        'config_file_list',
        nargs='+',
        type=str,
        help='Name of config file (eg: person_v1 for config/person_v1.json/yaml), or full path if it is not in config folder, you can pass multiple config files and run them all sequentially'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-r', '--recover',
        action='store_true',
        help='Continue running additional jobs even if a job fails'
    )

    # flag to continue if failed job
    parser.add_argument(
        '-n', '--name',
        type=str,
        default=None,
        help='Name to replace [name] tag in config file, useful for shared config file'
    )

    parser.add_argument(
        '-l', '--log',
        type=str,
        default=None,
        help='Log file to write output to'
    )

    parser.add_argument(
        '--tpu_cores',
        type=int,
        default=None,
        help='TPU cores for multi-core data-parallel training (overrides train.tpu_num_cores and AITK_TPU_CORES). Ignored off-TPU.'
    )
    args = parser.parse_args()

    config_file_list = args.config_file_list

    # Multi-core TPU: spawn one process per core before touching any job.
    # Single-core / CUDA / CPU path below is untouched.
    tpu_cores = _resolve_tpu_cores(config_file_list, args.tpu_cores)
    if tpu_cores > 1:
        try:
            from toolkit.xla_utils import is_xla_available, has_tpu
            tpu_visible = is_xla_available() and has_tpu()
        except Exception:
            tpu_visible = False
        if tpu_visible:
            print_acc(f"Spawning {tpu_cores} TPU worker processes")
            import torch_xla.distributed.xla_multiprocessing as xmp
            xmp.spawn(_tpu_worker, args=(config_file_list, args), nprocs=tpu_cores, start_method='spawn')
            return
        else:
            print_acc(
                f"tpu_num_cores={tpu_cores} requested but no TPU is visible; "
                f"running single-process."
            )

    _run_configs(config_file_list, args)


if __name__ == '__main__':
    main()
