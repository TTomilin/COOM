from argparse import Namespace

from coom.doom.env.extended.seek_and_slay_impl import SeekAndSlayImpl


class ChainsawImpl(SeekAndSlayImpl):

    def __init__(self, args: Namespace, task: str, task_id: int, num_tasks=1):
        super().__init__(args, task, task_id, num_tasks)
