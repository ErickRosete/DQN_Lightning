from setuptools import setup, find_packages

setup(
      name='dqn_lightning',
      version='0.0.1',
      description='Deep Q Network using PyTorch Lightning',
      author='Erick Rosete Beas',
      author_email='erickrosetebeas@hotmail.com',
      url='https://github.com/ErickRosete/DQN_Lightning',
      packages=find_packages(),
      install_requires=[
                'gym(==0.18.3)',
                'wandb(==0.10.32)',
                'pytorch-lightning(==1.3.6)']
     )
