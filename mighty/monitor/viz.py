import os
import sys
import time
from collections import defaultdict
from typing import Union, List

import numpy as np
import visdom

from mighty.monitor.batch_timer import timer


class VisdomMighty(visdom.Visdom):
    def __init__(self, env: str = "main"):
        port = int(os.environ.get('VISDOM_PORT', 8097))
        server = os.environ.get('VISDOM_SERVER', 'http://localhost')
        env = env.replace('_', '-')  # visdom things
        try:
            super().__init__(env=env, server=server, port=port,
                             username=os.environ.get('VISDOM_USER', None),
                             password=os.environ.get('VISDOM_PASSWORD', None),
                             raise_exceptions=True)
        except ConnectionError as error:
            tb = sys.exc_info()[2]
            raise ConnectionError("Start Visdom server with "
                                  "'python -m visdom.server' command."
                                  ).with_traceback(tb)
        print(f"Monitor is opened at {self.server}:{self.port}. "
              f"Choose environment '{self.env}'.")
        self.timer = timer
        self.legends = defaultdict(list)
        self.with_markers = False
        self.register_comments_window()

    def register_comments_window(self):
        txt_init = "Enter comments:"
        win = 'comments'

        def type_callback(event):
            if event['event_type'] == 'KeyPress':
                curr_txt = event['pane_data']['content']
                if event['key'] == 'Enter':
                    curr_txt += '<br>'
                elif event['key'] == 'Backspace':
                    curr_txt = curr_txt[:-1]
                elif event['key'] == 'Delete':
                    curr_txt = txt_init
                elif len(event['key']) == 1:
                    curr_txt += event['key']
                self.viz.text(curr_txt, win='comments')

        self.text(txt_init, win=win)
        self.register_event_handler(type_callback, win)

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
