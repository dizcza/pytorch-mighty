"""
Visdom server
-------------

.. autosummary::
    :toctree: toctree/monitor

    VisdomMighty
"""

import os
import sys
import time
from collections import defaultdict

import numpy as np
import visdom

from mighty.monitor.batch_timer import timer
from mighty.utils.constants import VISDOM_LOGS_DIR


class VisdomMighty(visdom.Visdom):
    """
    A Visdom server that updates measures in online fashion.

    Parameters
    ----------
    env : str, optional
        Environment name.
        Default: "main"
    offline : bool, optional
        Online (False) or offline (True) mode.
        Default: False
    """
    def __init__(self, env="main", offline=False):
        port = int(os.environ.get('VISDOM_PORT', 8097))
        server = os.environ.get('VISDOM_SERVER', 'http://localhost')
        base_url = os.environ.get('VISDOM_BASE_URL', '/')
        env = env.replace('_', '-')  # visdom things
        log_to_filename = None
        if offline:
            VISDOM_LOGS_DIR.mkdir(exist_ok=True)
            log_to_filename = VISDOM_LOGS_DIR / f"{env}.log"
        try:
            super().__init__(env=env, server=server, port=port,
                             username=os.environ.get('VISDOM_USER', None),
                             password=os.environ.get('VISDOM_PASSWORD', None),
                             log_to_filename=log_to_filename,
                             offline=offline,
                             base_url=base_url,
                             raise_exceptions=True)
        except ConnectionError as error:
            tb = sys.exc_info()[2]
            raise ConnectionError("Start Visdom server with "
                                  "'python -m visdom.server' command."
                                  ).with_traceback(tb)
        self.timer = timer
        self.legends = defaultdict(list)
        self.with_markers = False

        if offline:
            print(f"Visdom logs are saved in {log_to_filename}")
        else:
            url = f"{self.server}:{self.port}{self.base_url}"
            print(f"Monitor is opened at {url}. "
                  f"Choose environment '{self.env}'.")
            self._register_comments_window()

    def _register_comments_window(self):
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

    def line_update(self, y, opts, name=None):
        """
        Appends `y` axis value to the plot. The `x` axis value will be
        extracted from the global timer.

        Parameters
        ----------
        y : float or list of float or torch.Tensor
            The Y axis value.
        opts : dict
            Visdom plot `opts`.
        name : str or None, optional
            The label name of this plot. Used when a plot has a legend.
            Default: None

        """
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

    def log(self, text, timestamp=True):
        """
        Log the text.

        Parameters
        ----------
        text : str
            Text
        timestamp : bool, optional
            Prepend date timestamp (True) or not.
            Default: True
        """
        if timestamp:
            text = f"{time.strftime('%Y-%b-%d %H:%M')} {text}"
        self.text(text, win='log', append=self.win_exists(win='log'))
