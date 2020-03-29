# pytorch-mighty

The Mighty Monitor Trainer for your pytorch models. Powered by [Visdom](https://github.com/facebookresearch/visdom).

Requires Python 3.6+

### Quick start

1. Install [pytorch](https://pytorch.org/)
2. `$ pip install pytorch-mighty`
3. `$ python -m visdom.server -port 8097` - start visdom server on port 8097
4. In a separate terminal, run `python examples.py`
5. Navigate to http://localhost:8097 to see the training progress.
6. Check-out more examples on [http://85.217.171.57:8097](http://85.217.171.57:8097/). Give your browser a few minutes to parse the json data.


### Articles, implemented in the package

1. Fong, R. C., & Vedaldi, A. (2017). Interpretable explanations of black boxes by meaningful perturbation.
    * Paper: https://arxiv.org/abs/1704.03296
    * Used in: [`trainer/mask.py`](mighty/trainer/mask.py)

2. Belghazi, M. I., Baratin, A., Rajeswar, S., Ozair, S., Bengio, Y., Courville, A., & Hjelm, R. D. (2018). Mine: mutual information neural estimation.
    * Paper: https://arxiv.org/abs/1801.04062
    * Used in: [`monitor/mutual_info/neural_estimation.py`](mighty/monitor/mutual_info/neural_estimation.py)

3. Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information.
    * Paper: https://arxiv.org/abs/1208.4475
    * Used in: [`monitor/mutual_info/npeet.py`](mighty/monitor/mutual_info/npeet.py)
    * Original source code: https://github.com/gregversteeg/NPEET

4. Ince, R. A., Giordano, B. L., Kayser, C., Rousselet, G. A., Gross, J., & Schyns, P. G. (2017). A statistical framework for neuroimaging data analysis based on mutual information estimated via a gaussian copula. Human brain mapping, 38(3), 1541-1573.
    * Paper: http://dx.doi.org/10.1002/hbm.23471
    * Used in [`monitor/mutual_info/gcmi.py`](mighty/monitor/mutual_info/gcmi.py)
    * Original source code: https://github.com/robince/gcmi


### Projects that use pytorch-mighty

* [MCMC_BinaryNet](https://github.com/dizcza/MCMC_BinaryNet) - Markov Chain Monte Carlo binary networks optimization.
* [EmbedderSDR](https://github.com/dizcza/EmbedderSDR) - encode images into binary Sparse Distributed Representation ([SDR](https://discourse.numenta.org/t/sparse-distributed-representations/2150)).
