from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='baselines',
      packages=[package for package in find_packages()
                if package.startswith('baselines')],
      install_requires=[
      #    'gym[mujoco,atari,classic_control,robotics]',
         'gym[classic_control]',
         'scipy',
         'tqdm',
         'joblib',
         'zmq',
         'dill',
         'progressbar2',
         'mpi4py',
         'cloudpickle',
         'tensorflow==1.4.0',
         'click',
         'matplotlib'
      ],
      description='ETC using Deep RL code, based on OpenAI Gym',
      author='Jia-Jie Zhu',
      url='https://github.com/jj-zhu')
