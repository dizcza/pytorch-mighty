import os
import time
from collections import defaultdict
from typing import Union, List, Iterable

import numpy as np
import visdom

from mighty.monitor.batch_timer import timer


class VisdomMighty(visdom.Visdom):
    def __init__(self, env: str = "main"):
        port = int(os.environ.get('VISDOM_PORT', 8097))
        server = os.environ.get('VISDOM_SERVER', 'http://localhost')
        env = env.replace('_', '-')  # visdom things
        super().__init__(env=env, server=server, port=port,
                         username=os.environ.get('VISDOM_USER', None),
                         password=os.environ.get('VISDOM_PASSWORD', None))
        print(f"Monitor is opened at {self.server}:{self.port}. "
              f"Choose environment '{self.env}'.")
        self.timer = timer
        self.legends = defaultdict(list)
        self.with_markers = False

    def line_update(self, y: Union[float, List[float]], opts: dict, name=None):
        y = np.array([y])
        n_lines = y.shape[-1]
        if n_lines == 0:
            return
        if y.ndim > 1 and n_lines == 1:
            # visdom expects 1d array for a single line plot
            y = y[0]
        x = np.full_like(y, self.timer.epoch_progress(), dtype=np.float32)
        # hack to make window names consistent if the user forgets to specify
        # the title
        win = opts.get('title', str(opts))
        if self.with_markers:
            opts['markers'] = True
            opts['markersize'] = 7
        self.line(Y=y, X=x, win=win, opts=opts, update='append', name=name)
        if name is not None:
            self.update_window_opts(win=win, opts=dict(legend=[], title=win))

    def log(self, text: str):
        self.text(f"{time.strftime('%Y-%b-%d %H:%M')} {text}",
                  win='log', append=self.win_exists(win='log'))
