from transformers import PrinterCallback, ProgressCallback
from transformers.integrations import TensorBoardCallback, rewrite_logs
from transformers.integrations.integration_utils import logger


def rewrite_model_logs(logs):
    new_logs = logs.copy()
    model_logs = new_logs.pop("model_logs", None)
    if model_logs is not None:
        assert isinstance(model_logs, dict)
        new_logs.update(model_logs)
    return new_logs


def on_log_tensorboard(self, args, state, control, logs=None, **kwargs):
    if not state.is_world_process_zero:
        return

    if self.tb_writer is None:
        self._init_summary_writer(args)

    if self.tb_writer is not None:
        logs = rewrite_model_logs(logs)
        logs = rewrite_logs(logs)
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                self.tb_writer.add_scalar(k, v, state.global_step)
            elif isinstance(v, dict):
                self.tb_writer.add_scalars(k, v, state.global_step)
            else:
                logger.warning(
                    "Trainer is attempting to log a value of "
                    f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                    "This invocation of Tensorboard's writer.add_scalar() "
                    "is incorrect so we dropped this attribute."
                )
        self.tb_writer.flush()

def on_log_printer(self, args, state, control, logs=None, **kwargs):
    _ = logs.pop("total_flos", None)
    _ = logs.pop("model_logs", None)
    if state.is_local_process_zero:
        print(logs)

def on_log_progress(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)
        _ = logs.pop("model_logs", None)
        self.training_bar.write(str(logs))


def replace_callbacks_on_log():
    TensorBoardCallback.on_log = on_log_tensorboard
    PrinterCallback.on_log = on_log_printer
    ProgressCallback.on_log = on_log_progress

