# Reinforcement Learning for Event-Triggered Control

This is the implementation for the algorithm introduced in [this paper](https://arxiv.org/abs/1809.05152). The RL method implementation is based on the OpenAI Baselines framework. If you use this code for research, please kindly cite the paper:
```
@article{baumann2018deep,
  title={Deep Reinforcement Learning for Event-Triggered Control},
  author={Baumann, Dominik and Zhu, Jia-Jie and Martius, Georg and Trimpe, Sebastian},
  journal={arXiv preprint arXiv:1809.05152},
  year={2018}
}
```
## How to run the code
To run the simulations, Python version 3.5 or higher is required.


To install necessary dependencies (such as [OpenAI Gym](https://gym.openai.com/), [Baselines](https://github.com/openai/baselines)), start with

```python
pip install -e .

```
To run the pendulum swing-up simulation, run the script 
```bash
run_etc.sh
```

## Copyright
Copyright (c) 2018 Jia-Jie Zhu, Max Planck Institute for Intelligent Systems

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

PLEASE UNDERSTAND I MIGHT NOT BE ABLE TO PROVIDE SUPPORT FOR THE CODE IN A TIMELY MANNER