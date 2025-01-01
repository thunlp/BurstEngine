import functools
import json
import os
import shutil
import time
from typing import List

import bmtrain as bmt
import torch

from .log import logger


def rename_if_exists(file_path):
    if not os.path.exists(file_path):
        return
    timestamp = time.strftime("%Y%m%d%H%M%S")
    file_dir, file_name = os.path.split(file_path)
    file_root, file_ext = os.path.splitext(file_name)
    new_file_name = f"{file_root}_bak_{timestamp}{file_ext}"
    new_file_path = os.path.join(file_dir, new_file_name)
    try:
        os.rename(file_path, new_file_path)
        logger.info(f"File '{file_name}' already exists. Renamed to '{new_file_name}'")
    except Exception as e:
        logger.warn(
            "rename file failed,file_path={file_path}, new_file_path={new_file_path},err={err}".format(
                file_path=file_path, new_file_path=new_file_path, err=str(e)
            )
        )


def rename_if_exists_decorator(func):
    @functools.wraps(func)
    def wrapper(file_path, *args, **kwargs):
        rename_if_exists(file_path)
        return func(file_path, *args, **kwargs)

    return wrapper


@rename_if_exists_decorator
def bmt_save(file_path: str, model: torch.nn.Module, export_files: List[str] = None):
    bmt.save(model, file_path)
    if export_files is not None:
        export_files.append(file_path)


@rename_if_exists_decorator
def torch_save(file_path: str, obj: object, export_files: List[str] = None):
    torch.save(obj, file_path)
    if export_files is not None:
        export_files.append(file_path)


@rename_if_exists_decorator
def json_save(file_path: str, obj: object, export_files: List[str] = None):
    with open(file_path, "w") as data_f:
        json.dump(obj, data_f)
    if export_files is not None:
        export_files.append(file_path)


def export(
    model: torch.nn.Module, dataloader, optimizer: bmt.optim.AdamOffloadOptimizer, global_step, args, final_save=False
):
    """
    一次 ckpt 保存：
    /{args.save}/
        ├── {save_name}-{global_step}.rank-0.opt
        ├── {save_name}-{global_step}.rank-n.opt
        ├── job_{job_id}_ckpt_{global_step}/  # checkpoint 导出为模型版本时，job_{job_id}_ckpt_{global_step}/ 路径下文件会一起导出，创建一个模型组版本
            ├── config.json
            ├── vocabs.txt
            ├── {args.save_name}-{global_step}.pt
            ├── {args.save_name}-{global_step}.data
            ├── {args.save_name}-{global_step}.data.json
            └── {args.save_name}-{global_step}.success

    """
    platform_cfg = get_platform_cfg()
    export_model_dir = (
        args.save_model if final_save else platform_cfg.gen_export_ckpt_dir_for_step(args.save, global_step)
    )
    os.makedirs(export_model_dir, exist_ok=True)
    base_file_name = f"{args.save_name}-{global_step}" if global_step > -1 else args.save_name
    logger.info(f"start to export ckpt, save_dir={export_model_dir}, file prefix={base_file_name}")
    export_files = []

    # model checkpoint
    bmt_save(
        file_path=os.path.join(export_model_dir, base_file_name + ".pt"),
        model=model,
        export_files=export_files,
    )

    # opt 文件仅用于项目内续训，不需要导出为模型版本文件
    if not final_save:
        grad_path = os.path.join(
            args.save,
            args.save_name + ("-%d.rank-%d.opt" % (global_step % (args.save_iters * 5), bmt.rank())),
        )
        torch.save(optimizer.state_dict(), grad_path)
        logger.info(f"Successfully save grad file: {grad_path}")

    all_states = dataloader.state_dict()
    if bmt.rank() == 0:
        # data checkpoint
        # rank 0 writes the dataloader state
        torch_save(
            file_path=os.path.join(export_model_dir, base_file_name + ".data"),
            obj=all_states,
            export_files=export_files,
        )
        # data checkpoint json
        # rank 0 writes the dataloader state into the json file
        data_p_json = {k: v for k, v in all_states.items()}
        for k in data_p_json:
            data_p_json[k] = {k_of_v: data_p_json[k][k_of_v].tolist() for k_of_v in data_p_json[k]}
        json_save(
            file_path=os.path.join(export_model_dir, base_file_name + ".data.json"),
            obj=data_p_json,
            export_files=export_files,
        )
        # config 和 vocabs 和模型文件一起存储
        model_cfg_path = os.path.join(export_model_dir, "config.json")
        model_vocab_path = os.path.join(export_model_dir, "vocabs.txt")
        export_files.extend([model_cfg_path, model_vocab_path])
        shutil.copy(args.model_config, model_cfg_path)
        shutil.copy(args.vocab, model_vocab_path)
        # 存储完所有文件后调用
        platform_cfg.finalize_model_save(export_model_dir, base_file_name)
        logger.info(f"Successfully save model files!  {export_files}")
    del all_states
    return export_model_dir
