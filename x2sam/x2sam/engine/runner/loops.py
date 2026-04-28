import logging
import time
from typing import Dict, Optional, Sequence, Union

from mmengine.runner import IterBasedTrainLoop
from mmengine.runner.loops import IterBasedTrainLoop, _InfiniteDataloaderIterator
from torch.utils.data import DataLoader

from x2sam.utils.logging import print_log


# Refer to https://github.com/open-mmlab/mmengine/pull/1548
class X2SamInfiniteDataloaderIterator(_InfiniteDataloaderIterator):
    def skip_iter(self, iters: int) -> None:
        if iters <= 0:
            return

        # first try to skip iters by sampler, if failed, then skip iters by iterator
        sampler = next(
            (
                candidate
                for candidate in (
                    getattr(self._dataloader, "sampler", None),
                    getattr(getattr(self._dataloader, "batch_sampler", None), "sampler", None),
                )
                if hasattr(candidate, "set_epoch")
            ),
            None,
        )
        if sampler is not None:
            num_iters_per_epoch = len(self._dataloader)
            epoch, step = divmod(iters, num_iters_per_epoch)
            can_reset_by_epoch = True
            try:
                sampler.set_epoch(epoch, step)
            except TypeError:
                if step == 0:
                    sampler.set_epoch(epoch)
                else:
                    can_reset_by_epoch = False
            if can_reset_by_epoch:
                self._epoch = epoch
                self._iterator = iter(self._dataloader)
                return

        for _ in range(iters):
            self._next_data(skip_loading=True)

    def __next__(self) -> Sequence[dict]:
        return self._next_data()

    def _next_data(self, skip_loading=False) -> Sequence[dict]:
        data = None
        try:
            if skip_loading:
                self._iterator._next_index()
            else:
                data = next(self._iterator)
        except StopIteration:
            print_log(
                "Reach the end of the dataloader, it will be "
                "restarted and continue to iterate. It is "
                "recommended to use "
                "`mmengine.dataset.InfiniteSampler` to enable the "
                "dataloader to iterate infinitely.",
                logger="current",
                level=logging.WARNING,
            )
            self._epoch += 1
            if hasattr(self._dataloader, "sampler") and hasattr(self._dataloader.sampler, "set_epoch"):
                # In case the` _SingleProcessDataLoaderIter` has no sampler,
                # or data loader uses `SequentialSampler` in Pytorch.
                self._dataloader.sampler.set_epoch(self._epoch)

            elif hasattr(self._dataloader, "batch_sampler") and hasattr(
                self._dataloader.batch_sampler.sampler, "set_epoch"
            ):
                # In case the` _SingleProcessDataLoaderIter` has no batch
                # sampler. batch sampler in pytorch warps the sampler as its
                # attributes.
                self._dataloader.batch_sampler.sampler.set_epoch(self._epoch)
            time.sleep(30)  # Prevent possible deadlock during epoch transition
            self._iterator = iter(self._dataloader)
            data = next(self._iterator)
        return data


class X2SamIterBasedTrainLoop(IterBasedTrainLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader_iterator = X2SamInfiniteDataloaderIterator(self.dataloader)

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook("before_train")
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook("before_train_epoch")
        if self._iter > 0:
            resume_skip_iters = 0
            if hasattr(self.runner.optim_wrapper, "pop_resume_skip_iters"):
                resume_skip_iters = self.runner.optim_wrapper.pop_resume_skip_iters()
            total_skip_iters = self._iter + resume_skip_iters
            print_log(
                f"Advance dataloader {total_skip_iters} steps to skip data that has already been trained",
                logger="current",
                level=logging.WARNING,
            )
            self.dataloader_iterator.skip_iter(total_skip_iters)

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (
                self.runner.val_loop is not None
                and self._iter >= self.val_begin
                and (self._iter % self.val_interval == 0 or self._iter == self._max_iters)
            ):
                self.runner.val_loop.run()

        self.runner.call_hook("after_train_epoch")
        self.runner.call_hook("after_train")
        return self.runner.model


class TrainLoop(X2SamIterBasedTrainLoop):
    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        max_iters: Optional[int] = None,
        max_epochs: Union[int, float] = None,
        **kwargs,
    ) -> None:
        if max_iters is None and max_epochs is None:
            raise RuntimeError("Please specify the `max_iters` or " "`max_epochs` in `train_cfg`.")
        elif max_iters is not None and max_epochs is not None:
            raise RuntimeError("Only one of `max_iters` or `max_epochs` can " "exist in `train_cfg`.")
        else:
            if max_iters is not None:
                iters = int(max_iters)
                assert iters == max_iters, "`max_iters` should be a integer " f"number, but get {max_iters}"
            elif max_epochs is not None:
                if isinstance(dataloader, dict):
                    diff_rank_seed = runner._randomness_cfg.get("diff_rank_seed", False)
                    dataloader = runner.build_dataloader(dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
                iters = max_epochs * len(dataloader)
            else:
                raise NotImplementedError

        print_log(f"Training max_iters: {iters}.", logger="current")
        super().__init__(runner=runner, dataloader=dataloader, max_iters=iters, **kwargs)

    @staticmethod
    def _get_batch_modality(data_batch: Dict) -> str:
        data_samples = data_batch.get("data_samples", None)
        task_names = getattr(data_samples, "task_names", None)
        if task_names:
            modalities = {task_name.split("_", 1)[0] for task_name in task_names if task_name}
            if len(modalities) != 1:
                raise RuntimeError(f"Mixed task names in the same batch are not supported: {task_names}")
            modality = next(iter(modalities))
            if modality in {"img", "vid"}:
                return modality

        data_dict = data_batch.get("data_dict", {})
        if "pixel_values_videos" in data_dict:
            return "vid"
        return "img"

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        if hasattr(self.runner.optim_wrapper, "set_current_modality"):
            modality = self._get_batch_modality(data_batch)
            self.runner.optim_wrapper.set_current_modality(modality)
        super().run_iter(data_batch)
