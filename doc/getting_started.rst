===============
Getting started
===============

Installation
------------

1. Install `PyTorch <https://pytorch.org/>`_ with your favourite method.

   For example, to install CPU version of PyTorch using conda, run

   .. code-block:: sh

    conda install pytorch torchvision cpuonly -c pytorch

2. Install pytorch-mighty simply via pip:

   .. code-block:: sh

    pip install pytorch-mighty


Start a Visdom server
---------------------

Before running any scripts, make sure you've started a `Visdom
<https://github.com/facebookresearch/visdom>`_ server (installed
automatically) in a separate terminal window:

.. code-block:: sh

    python -m visdom.server

You need to run this command only once.


Train a model
-------------

The minimal code to train a linear AutoEncoder:

.. code-block:: python

    import torch
    import torch.nn as nn
    from torchvision.datasets import MNIST

    from mighty.models import AutoencoderLinear
    from mighty.monitor.monitor import MonitorLevel
    from mighty.trainer import TrainerAutoencoder
    from mighty.utils.data import DataLoader

    model = AutoencoderLinear(784, 128)
    data_loader = DataLoader(dataset_cls=MNIST)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = TrainerAutoencoder(model,
                                 criterion=criterion,
                                 data_loader=data_loader,
                                 optimizer=optimizer)
    # trainer.restore()  # uncomment to restore the saved state
    trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGNAL_TO_NOISE)
    trainer.train(n_epochs=10, mutual_info_layers=0)

It will print the input model

.. code-block::

    AutoencoderLinear(
      (encoder): Sequential(
        (0): Linear(in_features=784, out_features=128, bias=True)
        (1): ReLU(inplace=True)
      )
      (decoder): Linear(in_features=128, out_features=784, bias=True)
    )

and start publishing the progress status to a Visdom sever, which you run
previously. Navigate to http://localhost:8097 to see the training progress.

For the rest training examples, refer to `examples.py
<https://github.com/dizcza/pytorch-mighty/blob/master/examples.py>`_.

Other demo results can be found at http://85.217.171.57:8096/. Give your
browser a few minutes to parse the json data.


Remote monitoring
~~~~~~~~~~~~~~~~~

Instead of running a Visdom server locally, you can set up a training to push
status updates to a remove Visdom server by setting environment keys in your
python script:

.. code-block:: python

    os.environ['VISDOM_SERVER'] = "http://myweb-service.com"
    os.environ['VISDOM_PORT'] = '8097'


Additionally, you can provide an authentication mechanism:

.. code-block:: python

    os.environ['VISDOM_USER'] = 'user'
    os.environ['VISDOM_PASSWORD'] = 'password'
