import os

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from stormer.data.multi_step_datamodule import MultiStepDataRandomizedModule
from stormer.models.iterative_module import GlobalForecastIterativeModule
from stormer.models.hub.stormer import Block, WeatherEmbedding

from lightning.pytorch.loggers.wandb import WandbLogger


class CustomCLI(LightningCLI):
    def _instantiate_trainer(self, config, callbacks):
        # 复用 LightningCLI 的 trainer 实例化逻辑，同时插入本项目需要的回调与策略配置。
        key = "callbacks"
        if key in config:
            if config[key] is None:
                config[key] = []
            elif not isinstance(config[key], list):
                config[key] = [config[key]]
            config[key].extend(callbacks)
            if key in self.trainer_defaults:
                value = self.trainer_defaults[key]
                config[key] += value if isinstance(value, list) else [value]
            if self.save_config_callback and not config.get("fast_dev_run", False):
                config_callback = self.save_config_callback(
                    self._parser(self.subcommand),
                    self.config.get(str(self.subcommand), self.config),
                    **self.save_config_kwargs,
                )
                config[key].append(config_callback)
        else:
            rank_zero_warn(
                f"The `{self.trainer_class.__qualname__}` class does not expose the `{key}` argument so they will"
                " not be included."
            )

        if config["strategy"] == "fsdp":
            # 当用户在配置中指定 fsdp 时，替换成完整的 FSDPStrategy 实例。
            # - SHARD_GRAD_OP: 梯度分片，降低显存占用
            # - activation_checkpointing_policy / auto_wrap_policy: 仅对关键模块启用
            fsdp_strategy = FSDPStrategy(
                sharding_strategy="SHARD_GRAD_OP",
                activation_checkpointing_policy={Block, WeatherEmbedding},
                auto_wrap_policy={Block, WeatherEmbedding},
            )
            config["strategy"] = fsdp_strategy

        return self.trainer_class(**config)


def main():
    # 通过 LightningCLI 统一管理:
    # 1) 模型/数据模块实例化
    # 2) YAML 配置解析（omegaconf 模式）
    # 3) 训练器参数与命令行参数融合
    cli = CustomCLI(
        model_class=GlobalForecastIterativeModule,
        datamodule_class=MultiStepDataRandomizedModule,
        seed_everything_default=42,
        save_config_callback=SaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=False,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    # 注入经纬度（用于纬度加权指标）与标准化/反标准化变换（用于多步滚动预测）。
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_transforms(*cli.datamodule.get_transforms())

    # 训练区间与验证 lead time 由数据模块配置提供，交给模型用于验证阶段 roll-out。
    cli.model.set_base_intervals_and_lead_times(
        cli.datamodule.hparams.list_train_intervals,
        cli.datamodule.hparams.val_lead_times,
    )

    # 将 checkpoint 保存路径收拢到 default_root_dir/<run_name>/checkpoints，便于断点续训和管理。
    logger_name = cli.trainer.logger._name
    for i in range(len(cli.trainer.callbacks)):
        if isinstance(cli.trainer.callbacks[i], ModelCheckpoint):
            cli.trainer.callbacks[i] = ModelCheckpoint(
                dirpath=os.path.join(
                    cli.trainer.default_root_dir, logger_name, "checkpoints"
                ),
                monitor=cli.trainer.callbacks[i].monitor,
                mode=cli.trainer.callbacks[i].mode,
                save_top_k=cli.trainer.callbacks[i].save_top_k,
                save_last=cli.trainer.callbacks[i].save_last,
                verbose=cli.trainer.callbacks[i].verbose,
                filename=cli.trainer.callbacks[i].filename,
                auto_insert_metric_name=cli.trainer.callbacks[
                    i
                ].auto_insert_metric_name,
            )

    # 重新创建 WandbLogger，保证日志目录与 checkpoint 目录层次一致。
    cli.trainer.logger = WandbLogger(
        name=logger_name,
        project=cli.trainer.logger._wandb_init["project"],
        save_dir=os.path.join(cli.trainer.default_root_dir, logger_name),
    )

    # 若存在 last.ckpt 则自动恢复训练，否则从头开始。
    if os.path.exists(
        os.path.join(
            cli.trainer.default_root_dir, logger_name, "checkpoints", "last.ckpt"
        )
    ):
        ckpt_resume_path = os.path.join(
            cli.trainer.default_root_dir, logger_name, "checkpoints", "last.ckpt"
        )
    else:
        ckpt_resume_path = None

    # 进入 Lightning 标准训练循环。
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=ckpt_resume_path)


if __name__ == "__main__":
    main()
